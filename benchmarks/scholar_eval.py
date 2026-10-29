"""Sampled DORIS replay: prepare, run (paid), and report (free).

python -m benchmarks.scholar_eval prepare --native PATH --out DIR
python -m benchmarks.scholar_eval run --out DIR
python -m benchmarks.scholar_eval report --out DIR
"""
import argparse
import asyncio
from collections import Counter
from copy import deepcopy
from dataclasses import asdict
from decimal import Decimal
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
from statistics import mean, median
from time import perf_counter
import httpx
from slingsearch import Config
from slingsearch.config import MODELS
from slingsearch.output import budget_results, evidence_tokens
from slingsearch.passages import representation
from slingsearch.rankers import make_ranker
from benchmarks.datasets import doris
from benchmarks.evaluate import bootstrap_delta


def lexical_order(question, candidates):
    """BM25 (k1=1.2, b=.75) fitted on this question's fixed candidate pool."""
    terms = lambda s: re.findall(r'\w+', s.casefold())
    docs = [Counter(terms(representation(c, 400))) for c in candidates]
    avg = mean(sum(d.values()) for d in docs) or 1
    df = Counter(t for d in docs for t in d)
    query = set(terms(question))
    scores = []
    for candidate, doc in zip(candidates, docs):
        length = sum(doc.values())
        score = sum(math.log(1 + (len(docs)-df[t]+.5)/(df[t]+.5)) *
                    doc[t]*2.2/(doc[t] + 1.2*(.25+.75*length/avg))
                    for t in query if doc[t])
        scores.append((score, candidate))
    return [c for _, c in sorted(scores, key=lambda pair: (-pair[0], pair[1].id))]


def evaluate(case, ids):
    # Linear summed aspect gain follows the upstream DORIS normal nDCG.
    # No exponential amplification of a sum containing many aspect judgments.
    if len(ids) != len(set(ids)) or set(ids) - case.query.qrels.keys():
        raise ValueError('Invalid ranked identities')
    gains = case.query.qrels
    def ndcg(k):
        dcg = sum(gains[d]/math.log2(i+2) for i,d in enumerate(ids[:k]))
        ideal = sum(g/math.log2(i+2) for i,g in enumerate(sorted(gains.values(), reverse=True)[:k]))
        return dcg/ideal if ideal else 0.0
    available = {a: max(s[a] for s in case.aspect_scores.values()) for a in case.aspect_text}
    def coverage(k):
        numerator = sum(max((case.aspect_scores[d][a] for d in ids[:k]), default=0) for a in available)
        return numerator/sum(available.values()) if sum(available.values()) else 0.0
    # Upstream binary threshold: average aspect judgment >= 1, not gain > 0.
    relevant = {d for d,g in gains.items() if g >= len(case.aspect_text)}
    recall = lambda k: len(relevant.intersection(ids[:k]))/len(relevant) if relevant else None
    return {'nDCG@5': ndcg(5), 'nDCG@10': ndcg(10), 'R@10': recall(10),
            'R@20': recall(20), 'aspect_coverage@5': coverage(5)}


def prepare(native, out, size=16, seed=20261029):
    out.mkdir(parents=True, exist_ok=True)
    if (out/'manifest.json').exists():
        raise ValueError('Manifest already exists; do not silently change an experiment')
    # Upstream create_60_query_dataset: avoids the 40 queries excluded from its 60-query track.
    frame = [1,8,10,16,23,28,33,37,44,46] + list(range(50,100))
    selected = sorted(random.Random(seed).sample(frame, size))
    manifest = {'dataset':'DORIS-MAE v1', 'source':'https://zenodo.org/records/8299749',
                'seed':seed,'size':size,'indices':selected,'sampling_frame':frame,
                'candidate_policy':'Full released pool; no label-based candidate selection',
                'candidate_tokens':400,'models':MODELS,'methods':['bm25','random','jev','decider','voyage'],
                'output_budget':1500, 'max_results':5,
                'labels':'Released AnnoGPT per-aspect scores; sum for linear nDCG; mean >= 1 for recall',
                'ranking_reservation_limit_usd':2.0,
                'prior_session_spend_estimate_usd':1.011467092,
                'paperfinding_status':'Official dataset gated: anonymous401, cached authenticated account403; access requested'}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    cases = doris.load(native, selected)
    doris.save_cases(cases, out/'frozen.json')
    manifest['snapshot_sha256'] = hashlib.sha256((out/'frozen.json').read_bytes()).hexdigest()
    manifest['candidate_counts'] = [len(c.query.candidates) for c in cases]
    from slingsearch.output import tokens
    lengths = [tokens(r.title+'\n'+r.abstract) for c in cases for r in c.query.candidates]
    manifest['text_audit'] = {'mean_original_estimated_tokens':mean(lengths),
                              'fraction_exceeding_document_budget':sum(n>400 for n in lengths)/len(lengths),
                              'mean_question_estimated_tokens':mean(tokens(c.query.question) for c in cases)}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps({'prepared_queries':len(cases),'candidates':sum(manifest['candidate_counts'])}),flush=True)


class SpendGuard(httpx.AsyncBaseTransport):
    def __init__(self, path, limit):
        self.path, self.limit = path, Decimal(str(limit))
        self.state = json.loads(path.read_text()) if path.exists() else {
            'requests':0,'reserved_usd':0,'reported_usd':0,'unreported_requests':0,
            'scope':'Only the experiment owning this ledger; reservations persist across restarts; no retries'}
        self.inner = httpx.AsyncHTTPTransport(retries=0)

    def save(self):
        self.path.write_text(json.dumps(self.state,indent=2))

    async def handle_async_request(self, request):
        payload = json.loads(request.content)
        if (request.url.host != 'openrouter.ai' or request.url.path not in
            ('/api/alpha/decisions','/api/v1/rerank') or payload.get('model') not in MODELS.values()):
            raise ValueError('Request outside experiment scope')
        cache_dir = self.path.parent/'response-cache'
        cache_dir.mkdir(exist_ok=True)
        digest = hashlib.sha256(request.url.path.encode()+request.content).hexdigest()
        cache = cache_dir/(digest+'.json')
        if cache.exists():
            saved = json.loads(cache.read_text())
            return httpx.Response(saved['status_code'],json=saved['body'])
        # Reserve a dollar allowance before each paid request. 3x chars-as-tokens
        # at the highest configured input rate plus minimum allowance per call.
        reserve = max(Decimal('.0002'), Decimal(len(request.content))*Decimal('.000000042')*3)
        if Decimal(str(self.state['reserved_usd'])) + reserve > self.limit:
            raise ValueError('Experiment spend reservation exhausted')
        self.state['reserved_usd'] += float(reserve)
        self.state['requests'] += 1
        self.save()
        response = await self.inner.handle_async_request(request)
        await response.aread()
        try:
            cost = (response.json().get('usage') or {}).get('cost')
            if cost is not None:
                self.state['reported_usd'] += float(cost)
            else:
                self.state['unreported_requests'] += 1
        except (ValueError,AttributeError):
            self.state['unreported_requests'] += 1
        self.save()
        if response.is_success:
            cache.write_text(json.dumps({'status_code':response.status_code,'body':response.json()}))
        return response

    async def aclose(self):
        await self.inner.aclose()


async def run(out, key):
    manifest = json.loads((out/'manifest.json').read_text())
    if hashlib.sha256((out/'frozen.json').read_bytes()).hexdigest() != manifest['snapshot_sha256']:
        raise ValueError('Frozen candidate snapshot changed')
    cases = doris.load_cases(out/'frozen.json')
    path = out/'rankings.jsonl'
    records = [json.loads(s) for s in path.read_text().splitlines()] if path.exists() else []
    completed = {(r['query_id'],r['method']) for r in records if r['success']}
    config = Config(api_keys={'openrouter':key},candidate_tokens=400,ranking_concurrency=32,timeout=60)
    guard = SpendGuard(out/'spend.json',manifest['ranking_reservation_limit_usd'])
    async with httpx.AsyncClient(transport=guard,timeout=60) as client:
        for case in cases:
            q = case.query
            # Identical text, question, and candidate set for every method.
            # No source scores or annotation gains enter ranker inputs or tie breaks.
            candidates = sorted(deepcopy(q.candidates),key=lambda c:c.id)
            text_hash = hashlib.sha256(json.dumps([q.question,[(r.id,representation(r,400)) for r in candidates]]).encode()).hexdigest()
            for method in manifest['methods']:
                if (q.id,method) in completed:
                    continue
                start = perf_counter()
                record = {'query_id':q.id,'method':method,'input_sha256':text_hash,
                          'candidate_count':len(candidates),
                          'ranking_concurrency':config.ranking_concurrency}
                try:
                    if method == 'bm25':
                        ranked = lexical_order(q.question,candidates)
                    elif method == 'random':
                        ranked = deepcopy(candidates)
                        random.Random(str(manifest['seed'])+q.id).shuffle(ranked)
                    else:
                        ranker = make_ranker(method,client,config)
                        ranked = await ranker.rank(q.question,candidates)
                        record.update(model=MODELS[method],resolved_models=sorted(ranker.resolved_models),
                                      usage=ranker.usage,reported_cost_usd=ranker.cost_usd)
                    ids = [r.id for r in ranked]
                    bounded = budget_results(ranked,max_results=5,token_budget=1500)
                    used = evidence_tokens(bounded)
                    record.update(success=True,result_ids=ids,scores={r.id:r.model_score for r in ranked},
                                  metrics=evaluate(case,ids),output_ids=[r.id for r in bounded],
                                  output_tokens=used,output_metrics=evaluate(case,[r.id for r in bounded]))
                except Exception as error:
                    record.update(success=False,error_type=type(error).__name__)
                    # No exception strings: HTTP exceptions may carry sensitive context.
                record['latency_s'] = perf_counter()-start
                with path.open('a') as f:
                    f.write(json.dumps(record)+'\n')
                print(json.dumps({'query_id':q.id,'method':method,'success':record['success'],
                                  'latency_s':round(record['latency_s'],2),
                                  'reported_usd':guard.state['reported_usd']}),flush=True)
    report(out)


def report(out):
    # Last attempt per query/method supersedes an earlier recorded failure.
    attempts = [json.loads(s) for s in (out/'rankings.jsonl').read_text().splitlines()]
    rows = list({(r['query_id'],r['method']):r for r in attempts}.values())
    manifest = json.loads((out/'manifest.json').read_text())
    # Paired common success set; report failures, never silently substitute a baseline.
    by = {method:{r['query_id']:r for r in rows if r['method']==method and r['success']}
          for method in manifest['methods']}
    common = set.intersection(*(set(v) for v in by.values()))
    summary = {}
    metrics = ['nDCG@5','nDCG@10','R@10','R@20','aspect_coverage@5']
    for method, records in by.items():
        paired = [records[q] for q in sorted(common)]
        summary[method] = {'successful':len(records),'paired_queries':len(common),
                           'failed':sum(r['method']==method and not r['success'] for r in rows)}
        if not paired:
            continue
        summary[method].update({m:mean(r['metrics'][m] for r in paired if r['metrics'][m] is not None)
                               for m in metrics})
        summary[method]['metric_query_counts'] = {
            m:sum(r['metrics'][m] is not None for r in paired) for m in metrics}
        summary[method]['latency_p50_s'] = median(r['latency_s'] for r in paired)
        timings = sorted(r['latency_s'] for r in paired)
        summary[method]['latency_p95_s'] = timings[min(len(timings)-1,math.ceil(.95*len(timings))-1)]
        summary[method]['reported_ranking_cost_per_1000_usd'] = (
            1000*mean(r.get('reported_cost_usd') or 0 for r in paired)
            if method in ('bm25','random') or all(r.get('reported_cost_usd') is not None for r in paired)
            else None)
        summary[method]['output_tokens_mean'] = mean(r['output_tokens'] for r in paired)
        summary[method]['output_nDCG@5'] = mean(r['output_metrics']['nDCG@5'] for r in paired)
        if method != 'bm25':
            summary[method]['paired_vs_bm25'] = bootstrap_delta(
                {q:records[q]['metrics']['nDCG@5'] for q in common},
                {q:by['bm25'][q]['metrics']['nDCG@5'] for q in common})
    pairwise = {}
    for left,right in [('voyage','jev'),('decider','jev'),('voyage','decider')]:
        if common:
            pairwise[left+'_vs_'+right] = bootstrap_delta(
                {q:by[left][q]['metrics']['nDCG@5'] for q in common},
                {q:by[right][q]['metrics']['nDCG@5'] for q in common})
    payload = {'summary':summary,'pairwise':pairwise,'paired_query_ids':sorted(common),
               'manifest':manifest,'spend':json.loads((out/'spend.json').read_text()) if (out/'spend.json').exists() else {}}
    (out/'results.json').write_text(json.dumps(payload,indent=2))
    lines = ['# Sampled DORIS-MAE ranking evaluation','',
             f'{len(common)} paired queries; full released candidate pools; identical 400-estimated-token document representations.',
             '', '| Method | nDCG@5 | nDCG@10 | Recall@10 | Aspect coverage@5 | Latency p50 (s) | Ranking USD/1000 |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for method,s in summary.items():
        if s.get('paired_queries'):
            cost=s['reported_ranking_cost_per_1000_usd']
            cost_text=f'{cost:.3f}' if cost is not None else 'unreported'
            lines.append(f"| {method} | {s['nDCG@5']:.4f} | {s['nDCG@10']:.4f} | {s['R@10']:.4f} | {s['aspect_coverage@5']:.4f} | {s['latency_p50_s']:.2f} | {cost_text} |")
    lines += ['', '## Interpretation and limits','',
              '- Ranking-only test: these are native dataset candidates, not live Exa or blended provider results. No provider-fusion or diversity win can be inferred.',
              '- Labels are released AnnoGPT judgments, not exhaustive independent human gold. Linear summed gains match upstream DORIS normal nDCG. Recall uses upstream mean-aspect-score >= 1.',
              '- Recall excludes queries with no papers above that threshold, following the upstream convention. Per-metric query counts are recorded in results.json.',
              '- Aspect coverage is a diagnostic: sum of best selected judgment per aspect divided by the same quantity over the whole pool. It can reward partial papers and must be read alongside relevance.',
              '- BM25 sees the same bounded document text as the models. Random is one fixed seeded permutation per query. Neither baseline uses labels.',
              '- Long abstracts are truncated to a shared 400 estimated tokens, preserving literal prefixes. The labels were assigned to full abstracts, so relevant evidence beyond the cutoff can penalize every method. This is a budget-constrained test, not an unbounded full-abstract result.',
              '- Table measures document ordering; results.json separately includes the actual five-result/1500-token output metrics.',
              '- Ranking cost/1000 extrapolates full native pools of roughly 100 papers, not the product default 20-candidate shortlist. It excludes retrieval and any aborted or failed requests; spend.json tracks all attempted requests separately.',
              '- Query-level paired bootstrap intervals and full rankings are saved in results.json and rankings.jsonl. Small-sample results are provisional.',
              '- Runtime concurrency was increased during the run (12 to 32); one question resumes cached judgments. Latency figures describe observed execution, not a controlled speed benchmark. See manifest runtime_note.',
              '- PaperFindingBench is a separate retrieval evaluation; do not describe this ranking-only run as a PaperFindingBench or provider-fusion result.',
              '', '## Reproduce','',
              'Free replay: `python -m benchmarks.scholar_eval report --out '+str(out)+'`',
              '', 'Paid run/resume: set OPENROUTER_API_KEY, then `python -m benchmarks.scholar_eval run --out '+str(out)+'`. Completed query/method pairs are reused. Reservations persist across restarts.']
    (out/'report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(payload['summary'],indent=2),flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action',choices=['prepare','run','report'])
    parser.add_argument('--native',type=Path)
    parser.add_argument('--out',type=Path,default=Path('benchmarks/reports/scholar-benchmarks/doris'))
    parser.add_argument('--size',type=int,default=16)
    args = parser.parse_args()
    if args.action == 'prepare':
        prepare(args.native,args.out,args.size)
    elif args.action == 'report':
        report(args.out)
    else:
        key = os.environ.get('OPENROUTER_API_KEY')
        if not key:
            raise ValueError('OPENROUTER_API_KEY is required')
        asyncio.run(run(args.out,key))


if __name__ == '__main__':
    main()
