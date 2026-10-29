"""Budgeted live adaptation of SAGE and official LitQA validation queries.

python -m benchmarks.academic_eval run   # private OpenRouter prompt
python -m benchmarks.academic_eval report
"""
import asyncio
from copy import deepcopy
from dataclasses import asdict
import getpass
import hashlib
import json
from pathlib import Path
from statistics import mean
from time import perf_counter
from urllib.parse import urlsplit
import httpx
from benchmarks.datasets import load_snapshot
from benchmarks.decision_budget import Transport, Budget
from benchmarks.evaluate import metrics, bootstrap_delta
from benchmarks.paperfinding_eval import metadata_id
from slingsearch import Config, Result
from slingsearch.fusion import fuse, normalize, normalize_doi
from slingsearch.output import budget_results, evidence_tokens
from slingsearch.passages import select
from slingsearch.providers import PROVIDERS
from slingsearch.rankers import make_ranker
from slingsearch.rankers.ensemble import fuse_orders

ROOT = Path('benchmarks/reports/academic-samples')
BUDGET = Path('benchmarks/reports/decision-audit')
PROVIDERS_USED = ['exa', 'parallel', 'openalex', 'semantic_scholar']


def lookup_id(result):
    pid = metadata_id(result)
    if pid:
        return pid
    host = urlsplit(result.url).hostname or ''
    if host in ('www.semanticscholar.org', 'semanticscholar.org', 'arxiv.org', 'www.biorxiv.org', 'aclanthology.org'):
        return 'URL:' + result.url
    return None


async def metadata(client, keys, ids):
    details = {}
    for start in range(0, len(ids), 100):
        batch = ids[start:start + 100]
        for attempt in range(3):
            await asyncio.sleep(1.1 if attempt == 0 else 3)
            response = await client.post('https://api.semanticscholar.org/graph/v1/paper/batch',
                headers={'x-api-key': keys['semantic_scholar']},
                params={'fields': 'title,year,publicationDate,abstract,externalIds,corpusId'}, json={'ids': batch})
            if response.status_code != 429:
                break
        if response.is_success:
            details.update({pid: paper for pid, paper in zip(batch, response.json()) if paper})
    return details


def aliases(paper):
    ids = {k.lower(): str(v) for k, v in (paper.get('externalIds') or {}).items() if v}
    if paper.get('paperId'): ids['semantic_scholar'] = paper['paperId']
    if paper.get('corpusId'): ids['corpusid'] = str(paper['corpusId'])
    return ids


async def collect(client, keys):
    queries = load_snapshot(ROOT / 'sample.json')
    completed = {q.id for q in load_snapshot(ROOT / 'retrieval.jsonl')} if (ROOT / 'retrieval.jsonl').exists() else set()
    gold_path = ROOT / 'gold-metadata.json'
    if not gold_path.exists():
        ids = sorted({pid.replace('semantic_scholar:', '').replace('corpusid:', 'CorpusId:') for q in queries for pid in q.qrels})
        gold_path.write_text(json.dumps(await metadata(client, keys, ids)))
    for q in queries:
        if q.id in completed: continue
        async def retrieve(name):
            adapter = PROVIDERS[name](client, keys[name], 'scholar')
            start = perf_counter()
            try:
                docs = await adapter.search(q.question, 10)
                status = {'success': True, 'count': len(docs), 'reported_cost_usd': adapter.cost_usd}
            except Exception as error:
                docs, status = [], {'success': False, 'error_type': type(error).__name__,
                    'http_status': getattr(getattr(error, 'response', None), 'status_code', None)}
            status['latency_s'] = perf_counter() - start
            q.retrieval[name] = status
            return docs
        raw = [normalize(r) for batch in await asyncio.gather(*(retrieve(n) for n in PROVIDERS_USED)) for r in batch]
        with (ROOT / 'raw-retrieved.jsonl').open('a') as f:
            f.write(json.dumps({'query_id': q.id, 'results': [asdict(r) for r in raw]}) + '\n')
        details = await metadata(client, keys, sorted({pid for r in raw if (pid := lookup_id(r))}))
        accepted, exclusions = [], 0
        for r in raw:
            paper = details.get(lookup_id(r))
            if paper:
                r.identifiers.update(aliases(paper))
                r.doi = r.doi or r.identifiers.get('doi')
                r.year = paper.get('year') or r.year
                r.abstract = paper.get('abstract') or r.abstract
                normalize(r)
            if q.id.startswith('litqa2_'):
                date = paper.get('publicationDate') if paper else None
                valid = date < '2024-10-17' if date else r.year is not None and r.year < 2024
                if not valid:
                    exclusions += 1
                    continue
            accepted.append(r)
        q.candidates = fuse(accepted)
        q.retrieval['audit'] = {'historical_date_exclusions': exclusions, 'unique_candidates': len(q.candidates),
            'metadata_resolved': len(details), 'metadata_requested': len({lookup_id(r) for r in raw if lookup_id(r)})}
        with (ROOT / 'retrieval.jsonl').open('a') as f: f.write(json.dumps(asdict(q)) + '\n')
        print(json.dumps({'query': q.id, 'phase': 'retrieve', 'candidates': len(q.candidates),
                          'providers': {n: q.retrieval[n]['success'] for n in PROVIDERS_USED}}), flush=True)


def gold_aliases(q, gold):
    mapping = {}
    for pid in q.qrels:
        mapping[pid] = pid
        lookup = pid.replace('semantic_scholar:', '').replace('corpusid:', 'CorpusId:')
        for key, value in aliases(gold.get(lookup, {})).items():
            alias = key + ':' + (normalize_doi(value) if key == 'doi' else value)
            mapping[alias] = pid
    return mapping


def score(q, ranked, gold):
    mapping = gold_aliases(q, gold)
    matched, seen = [], set()
    for r in ranked:
        identifiers = [r.id] + [k + ':' + (normalize_doi(v) if k == 'doi' else v) for k, v in r.identifiers.items()]
        if r.doi: identifiers.append('doi:' + normalize_doi(r.doi))
        pid = next((mapping[a] for a in identifiers if a in mapping), r.id)
        if pid in seen: continue
        seen.add(pid)
        matched.append(Result(pid, r.title, r.url))
    values = metrics(matched, q.qrels)
    for k in (5, 10, 20, 30):
        found = {r.id for r in matched[:k]} & q.qrels.keys()
        values[f'Hit@{k}'] = float(bool(found))
        values[f'Recall@{k}'] = len(found) / len(q.qrels)
        values[f'WeightedRecall@{k}'] = sum(q.qrels[d] for d in found) / sum(q.qrels.values())
    return values


async def rank(client, key):
    gold = json.loads((ROOT / 'gold-metadata.json').read_text())
    path = ROOT / 'rankings.jsonl'
    records = {(r['query_id'], r['method']): r for r in [json.loads(s) for s in path.read_text().splitlines()]} if path.exists() else {}
    for q in load_snapshot(ROOT / 'retrieval.jsonl'):
        pool = deepcopy(q.candidates[:50])
        for r in pool: r.passages = select(q.question, [r.abstract] if r.abstract else r.passages)
        singles = {name: sorted([r for r in pool if any(s.provider == name for s in r.sources)],
                     key=lambda r: min(s.rank for s in r.sources if s.provider == name)) for name in PROVIDERS_USED}
        runs = {'rrf': pool, **{'single:' + name: docs for name, docs in singles.items()}}
        methods = [*runs, 'jev_packed5', 'd1_packed5', 'voyage', *[name + '+voyage' for name in PROVIDERS_USED]]
        by_id = {r.id: r for r in pool}
        for method in methods:
            prior = records.get((q.id, method))
            if prior and prior['success']: continue
            backend = None
            start = perf_counter()
            try:
                if method in runs:
                    ranked = runs[method]
                    if method.startswith('single:') and not q.retrieval[method.split(':')[1]]['success']:
                        raise ValueError('Retrieval failed')
                else:
                    if '+' in method and not q.retrieval[method.split('+')[0]]['success']:
                        raise ValueError('Retrieval failed')
                    candidates = singles[method.split('+')[0]] if '+' in method else pool
                    backend = make_ranker('jev' if method == 'jev_packed5' else 'd1' if method == 'd1_packed5' else 'voyage',
                        client, Config(api_keys={'openrouter': key}, decision_batch_size=5, candidate_tokens=400))
                    ranked = await backend.rank(q.question, candidates) if candidates else []
                evidence = budget_results(ranked, 5, 1500)
                row = {'query_id': q.id, 'method': method, 'success': True,
                    'result_ids': [r.id for r in ranked], 'metrics': score(q, ranked, gold),
                    'product_metrics': score(q, evidence, gold), 'output_tokens': evidence_tokens(evidence)}
            except Exception as error:
                row = {'query_id': q.id, 'method': method, 'success': False, 'error_type': type(error).__name__}
            row.update(latency_s=perf_counter() - start, reported_cost_usd=backend.cost_usd if backend else 0,
                       usage=backend.usage if backend else {}, resolved_models=sorted(backend.resolved_models) if backend else [])
            records[q.id, method] = row
            with path.open('a') as f: f.write(json.dumps(row) + '\n')
            print(json.dumps({'query': q.id, 'phase': 'rank', 'method': method, 'success': row['success']}), flush=True)
        for method, members in {'jev_voyage': ['jev_packed5', 'voyage'], 'd1_voyage': ['d1_packed5', 'voyage'],
                                'jev_d1_voyage': ['jev_packed5', 'd1_packed5', 'voyage']}.items():
            if (q.id, method) in records and records[q.id, method]['success']: continue
            if not all(records[q.id, m]['success'] for m in members): continue
            ordered = fuse_orders([[by_id[pid] for pid in records[q.id, m]['result_ids']] for m in members], pool)
            evidence = budget_results(ordered, 5, 1500)
            row = {'query_id': q.id, 'method': method, 'success': True, 'result_ids': [r.id for r in ordered],
                'metrics': score(q, ordered, gold), 'product_metrics': score(q, evidence, gold),
                'output_tokens': evidence_tokens(evidence), 'latency_s': None,
                'reported_cost_usd': sum(records[q.id, m]['reported_cost_usd'] or 0 for m in members)}
            records[q.id, method] = row
            with path.open('a') as f: f.write(json.dumps(row) + '\n')


def report():
    queries = load_snapshot(ROOT / 'retrieval.jsonl')
    records = {(r['query_id'], r['method']): r for r in [json.loads(s) for s in (ROOT / 'rankings.jsonl').read_text().splitlines()]}
    groups = {'sage_short': [q for q in queries if q.id.startswith('sage_short')],
              'sage_open': [q for q in queries if q.id.startswith('sage_open')],
              'litqa2': [q for q in queries if q.id.startswith('litqa2')]}
    output = {'groups': {}, 'limitations': ['Live API adaptation, not official corpus leaderboard',
        'Only ten requested results per provider; single-provider Recall@30 is depth-limited',
        'SAGE reference-derived positives are incomplete; no title-only matching',
        'LitQA publication cutoff approximates but does not verify index insertion cutoff',
        'Metadata is pooled for identity and bounded document text before all ranking comparisons']}
    for group, qs in groups.items():
        methods = sorted({m for qid, m in records if qid in {q.id for q in qs}})
        summaries = {}
        for method in methods:
            rows = [records[q.id, method] for q in qs if (q.id, method) in records and records[q.id, method]['success']]
            summaries[method] = {'queries': len(rows), 'metrics': {k: mean(r['metrics'][k] for r in rows) for k in rows[0]['metrics']} if rows else {},
                'mean_reported_rank_cost_usd': mean(r['reported_cost_usd'] or 0 for r in rows) if rows else None}
        paired = {}
        for method in ['rrf', 'jev_packed5', 'd1_packed5', 'jev_voyage', 'd1_voyage', 'jev_d1_voyage']:
            for baseline in ['voyage', *[p + '+voyage' for p in PROVIDERS_USED]]:
                metric = 'WeightedRecall@10' if group == 'sage_open' else 'Hit@10' if group == 'sage_short' else 'Recall@30'
                left = {q.id: records[q.id, method]['metrics'][metric] for q in qs if records.get((q.id, method), {}).get('success')}
                right = {q.id: records[q.id, baseline]['metrics'][metric] for q in qs if records.get((q.id, baseline), {}).get('success')}
                if left.keys() & right.keys(): paired[method + ' vs ' + baseline] = bootstrap_delta(left, right)
        output['groups'][group] = {'available_queries': len(qs), 'summary': summaries, 'paired': paired,
            'provider_successes': {p: sum(q.retrieval[p]['success'] for q in qs) for p in PROVIDERS_USED},
            'by_domain': {domain: {'queries': len(sub), 'voyage': {k: (mean(values) if (values := [records[q.id, 'voyage']['metrics'][k] for q in sub if records.get((q.id, 'voyage'), {}).get('success')]) else None) for k in ['Hit@10', 'WeightedRecall@10']}}
                          for domain in sorted({q.retrieval['domain'] for q in qs if 'domain' in q.retrieval})
                          if (sub := [q for q in qs if q.retrieval.get('domain') == domain])}}
    output['snapshot_sha256'] = hashlib.sha256((ROOT / 'retrieval.jsonl').read_bytes()).hexdigest()
    (ROOT / 'results.json').write_text(json.dumps(output, indent=2))
    print(json.dumps({g: {m: s['metrics'].get('WeightedRecall@10' if g == 'sage_open' else 'Hit@10' if g == 'sage_short' else 'Recall@30') for m, s in v['summary'].items()} for g, v in output['groups'].items()}, indent=2))


async def run(key):
    # Expand the sub-experiment allowance, keeping the unchanged hard $5 session ceiling.
    Budget(BUDGET).edit(lambda d: d.update(new_limit_usd=3.0))
    keys = Config().api_keys
    async with httpx.AsyncClient(transport=Transport(BUDGET), timeout=60) as client:
        await collect(client, keys)
        await rank(client, key)
    report()


if __name__ == '__main__':
    import sys
    if sys.argv[1] == 'report': report()
    else: asyncio.run(run(getpass.getpass('OpenRouter key: ')))
