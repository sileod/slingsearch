"""Sampled PaperFindingBench retrieval/ranking and independent rubric evaluation.

Official source criteria, pinned GPT-4o judge, and corrected criterion-name matching.
This bounded five-result adaptation is not an official AstaBench leaderboard run.
"""
import argparse
import ast
import asyncio
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import random
from statistics import mean
from time import perf_counter
import httpx
from benchmarks.datasets import paperfinding, load_snapshot
from benchmarks.diversity import balanced_shortlist
from benchmarks.evaluate import bootstrap_delta
from benchmarks.scholar_eval import SpendGuard
from slingsearch import Config, Result
from slingsearch.fusion import fuse, normalize, title_key
from slingsearch.output import budget_results, evidence_tokens
from slingsearch.passages import representation, select
from slingsearch.providers import PROVIDERS
from slingsearch.rankers import make_ranker

PROVIDERS_USED = ['exa','openalex','semantic_scholar','parallel']
METHODS = ['single:'+p for p in PROVIDERS_USED]+['rrf','jev','decider','voyage','exa+jev','exa+voyage']


def prepare(native, out):
    out.mkdir(parents=True,exist_ok=True)
    if (out/'manifest.json').exists():
        return
    cases = sorted([c for c in paperfinding.load(native) if c.query_type=='semantic'],key=lambda c:c.query.id)
    selected = sorted(random.Random(20261029).sample(cases,12),key=lambda c:c.query.id)
    m = {'dataset':'PaperFindingBench validation','revision':'a600dc767f850385f4664772e3ba7a7f8be17d5e',
         'seed':20261029,'query_ids':[c.query.id for c in selected],
         'source_sha256':hashlib.sha256(Path(native).read_bytes()).hexdigest(),
         'methods':METHODS,'providers':PROVIDERS_USED,'per_provider':10,'shortlist':20,
         'candidate_tokens':400,'output_tokens':1500,'max_results':5,
         'cutoff':'2025-06-01','judge':'gpt-4o-2024-11-20',
         'judge_scope':'Independent official rubric/model; shared bounded document text; known positives grade3; no answer generation',
         'ranking_reservation_usd':.30,'judge_reservation_usd':2.50,
         'adaptation':'Five-result pooled evaluation, not official leaderboard. Estimated recall uses released normalizer. Criterion names case/underscore normalized; missing judgments fail, never silently dropped.'}
    (out/'manifest.json').write_text(json.dumps(m,indent=2))
    (out/'criteria.json').write_text(json.dumps({c.query.id:c.criteria for c in selected},indent=2))
    (out/'queries.jsonl').write_text(''.join(json.dumps(asdict(c.query))+'\n' for c in selected))


def metadata_id(r):
    ids=r.identifiers
    if ids.get('semantic_scholar'):return ids['semantic_scholar']
    if ids.get('arxiv'):return 'ARXIV:'+ids['arxiv']
    if r.doi:
        if r.doi.startswith('10.48550/arxiv.'):
            return 'ARXIV:'+r.doi.split('arxiv.',1)[1]
        return 'DOI:'+r.doi
    if ids.get('acl'):return 'ACL:'+ids['acl']
    if ids.get('pmid'):return 'PMID:'+ids['pmid']
    return None


async def collect(out, keys):
    if (out/'frozen.jsonl').exists():return
    qs=load_snapshot(out/'queries.jsonl')
    ledger=json.loads((out/'retrieval-spend.json').read_text()) if (out/'retrieval-spend.json').exists() else {'reported_retrieval_usd':0.,'parallel_requests':0,'parallel_estimated_usd':0.,
            'scope':'12 queries x 4 providers; S2 metadata enrichment; no retries beyond adapters'}
    prior={q.id:q for q in load_snapshot(out/'retrieval.jsonl')} if (out/'retrieval.jsonl').exists() else {}
    async with httpx.AsyncClient(timeout=45) as client:
        for index,q in enumerate(qs):
            if q.id in prior:
                qs[index]=prior[q.id]
                continue
            async def retrieve(name):
                p=PROVIDERS[name](client,keys[name],'scholar',year=(1900,2025))
                start=perf_counter()
                if name=='parallel':ledger['parallel_requests']+=1;ledger['parallel_estimated_usd']+=.005
                try:
                    rows=await p.search(q.question,10)
                    status={'success':True,'raw_count':len(rows),'reported_cost':p.cost_usd}
                except Exception as error:
                    rows=[];status={'success':False,'error_type':type(error).__name__,
                                    'http_status':getattr(getattr(error,'response',None),'status_code',None)}
                ledger['reported_retrieval_usd']+=p.cost_usd or 0
                status['latency_s']=perf_counter()-start
                q.retrieval[name]=status
                return rows
            batches=await asyncio.gather(*(retrieve(p) for p in PROVIDERS_USED))
            rows=[normalize(r) for batch in batches for r in batch]
            lookups=sorted({metadata_id(r) for r in rows if metadata_id(r)})
            details={}
            if lookups:
                await asyncio.sleep(2.2)
                response=await client.post('https://api.semanticscholar.org/graph/v1/paper/batch',
                    headers={'x-api-key':keys['semantic_scholar']},
                    params={'fields':'title,year,publicationDate,abstract,externalIds,corpusId'},
                    json={'ids':lookups})
                if response.is_success:
                    details={pid:p for pid,p in zip(lookups,response.json()) if p}
                q.retrieval['metadata']={'success':response.is_success,'http_status':response.status_code,
                                         'resolved':len(details),'requested':len(lookups)}
            accepted=[]
            audit=[]
            for r in rows:
                p=details.get(metadata_id(r))
                date=None
                if p:
                    r.year=p.get('year') or r.year
                    r.abstract=p.get('abstract') or r.abstract
                    r.identifiers.update({k.lower():str(v) for k,v in (p.get('externalIds') or {}).items() if v})
                    r.identifiers['corpusid']=str(p['corpusId'])
                    r.identifiers['semantic_scholar']=p['paperId']
                    r.doi=r.doi or r.identifiers.get('doi')
                    date=p.get('publicationDate')
                    normalize(r)
                valid = date<'2025-06-01' if date else r.year is not None and r.year<2025
                audit.append({'id':r.id,'title':r.title,'provider':r.sources[0].provider,
                              'year':r.year,'publication_date':date,'accepted':valid,
                              'reason':'before cutoff' if valid else 'after cutoff or date unverified'})
                if valid:accepted.append(r)
            q.candidates=accepted
            for name in PROVIDERS_USED:
                q.retrieval[name]['accepted_count']=sum(r.sources[0].provider==name for r in accepted)
            with (out/'retrieval.jsonl').open('a') as f:f.write(json.dumps(asdict(q))+'\n')
            with (out/'date-audit.jsonl').open('a') as f:f.write(json.dumps({'query_id':q.id,'rows':audit})+'\n')
            (out/'retrieval-spend.json').write_text(json.dumps(ledger,indent=2))
            print(json.dumps({'phase':'paperfinding_retrieval','query':q.id,
                              'providers':{p:q.retrieval[p]['accepted_count'] for p in PROVIDERS_USED},
                              'reported_plus_parallel_usd':ledger['reported_retrieval_usd']+ledger['parallel_estimated_usd']}),flush=True)
            if ledger['reported_retrieval_usd']+ledger['parallel_estimated_usd']>.30:
                raise ValueError('Retrieval spend stop')
            await asyncio.sleep(1)
    # Only mark frozen after all twelve have completed.
    (out/'frozen.jsonl').write_text(''.join(json.dumps(asdict(q))+'\n' for q in qs))
    m=json.loads((out/'manifest.json').read_text())
    m['snapshot_sha256']=hashlib.sha256((out/'frozen.jsonl').read_bytes()).hexdigest()
    (out/'manifest.json').write_text(json.dumps(m,indent=2))


async def rank(out,key):
    manifest=json.loads((out/'manifest.json').read_text()) if (out/'manifest.json').exists() else {}
    if manifest.get('snapshot_sha256') and hashlib.sha256((out/'frozen.jsonl').read_bytes()).hexdigest()!=manifest['snapshot_sha256']:
        raise ValueError('Frozen retrieval snapshot changed')
    qs=load_snapshot(out/'frozen.jsonl');path=out/'rankings.jsonl'
    rows=[json.loads(s) for s in path.read_text().splitlines()] if path.exists() else []
    completed={(r['query_id'],r['method']) for r in rows if r['success']}
    config=Config(api_keys={'openrouter':key},candidate_tokens=400,ranking_concurrency=16,timeout=60)
    guard=SpendGuard(out/'ranking-spend.json',.30)
    async with httpx.AsyncClient(transport=guard,timeout=60) as client:
        for q in qs:
            pool=fuse(q.candidates)
            for r in pool:r.passages=select(q.question,[r.abstract] if r.abstract else r.passages)
            shortlist=balanced_shortlist(pool,20)
            singles={p:sorted([r for r in pool if any(s.provider==p for s in r.sources)],
                       key=lambda r:min(s.rank for s in r.sources if s.provider==p)) for p in PROVIDERS_USED}
            for method in METHODS:
                if (q.id,method) in completed:continue
                record={'query_id':q.id,'method':method};start=perf_counter()
                try:
                    if method.startswith('single:'):ranked=singles[method.split(':')[1]]
                    elif method=='rrf':ranked=shortlist
                    else:
                        name=method.split('+')[-1]
                        candidates=singles['exa'] if method.startswith('exa+') else shortlist
                        if candidates:
                            ranker=make_ranker(name,client,config)
                            ranked=await ranker.rank(q.question,candidates)
                            record.update(usage=ranker.usage,reported_cost_usd=ranker.cost_usd,
                                          resolved_models=sorted(ranker.resolved_models))
                        else:ranked=[]
                    output=budget_results(ranked,max_results=5,token_budget=1500)
                    record.update(success=True,result_ids=[r.id for r in output],
                                  ranking_ids=[r.id for r in ranked],output_tokens=evidence_tokens(output),
                                  output=[asdict(r) for r in output])
                except Exception as error:
                    record.update(success=False,error_type=type(error).__name__)
                record['latency_s']=perf_counter()-start
                with path.open('a') as f:f.write(json.dumps(record)+'\n')
                print(json.dumps({'phase':'paperfinding_ranking','query':q.id,'method':method,
                                  'success':record['success'],'reported_usd':guard.state['reported_usd']}),flush=True)


def criterion_key(name):
    return ' '.join(name.replace('_',' ').casefold().split())


def parse_judgment(data,criteria):
    grades={'Perfectly Relevant':3,'Highly Relevant':2,'Somewhat Relevant':1,'Not Relevant':0}
    given={criterion_key(k):v for k,v in data['criteria'].items()}
    if any(criterion_key(c['name']) not in given for c in criteria):
        raise ValueError('Missing evaluator criterion')
    scores={c['name']:grades[given[criterion_key(c['name'])]['relevance']] for c in criteria}
    weighted=min(1.,sum(c['weight']*scores[c['name']]/3 for c in criteria))
    grade=0 if weighted<=.25 else 1 if weighted<=.67 else 2 if weighted<=.99 else 3
    return {'grade':grade,'criterion_scores':scores,'weighted':weighted,'raw':data}


def judge_prompt(source,criteria,doc):
    # Load only a literal string from the pinned public scorer source, never execute it.
    tree=ast.parse(Path(source).read_text())
    template=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign)
                  and any(isinstance(t,ast.Name) and t.id=='relevance_criteria_judgement_prompt_with_relevant_snippets_after' for t in n.targets))
    return template.format(criteria={'doc_id':doc.id,
        'criteria':json.dumps([{'name':c['name'],'description':c['description']} for c in criteria],indent=2),
        'document':representation(doc,400)})


async def judge(out,openrouter_key,source):
    import tiktoken
    manifest=json.loads((out/'manifest.json').read_text()) if (out/'manifest.json').exists() else {}
    checks=[('snapshot_sha256',out/'frozen.jsonl'),
            ('judge_prompt_source_sha256',Path(source)),('criteria_sha256',out/'criteria.json')]
    for name,path in checks:
        if manifest.get(name) and hashlib.sha256(path.read_bytes()).hexdigest()!=manifest[name]:
            raise ValueError('Frozen evaluator input changed: '+name)
    encoder=tiktoken.get_encoding('o200k_base')
    qs={q.id:q for q in load_snapshot(out/'frozen.jsonl')}
    criteria=json.loads((out/'criteria.json').read_text())
    rankings=list({(r['query_id'],r['method']):r for r in
                  [json.loads(s) for s in (out/'rankings.jsonl').read_text().splitlines()]}.values())
    path=out/'judgments.jsonl'
    completed={(r['query_id'],r['id']) for r in [json.loads(s) for s in path.read_text().splitlines()]
               if r.get('success')} if path.exists() else set()
    spend_path=out/'judge-spend.json'
    spend=json.loads(spend_path.read_text()) if spend_path.exists() else {
        'reserved_usd':0.,'estimated_actual_usd':0.,'requests':0,'prompt_tokens':0,'completion_tokens':0,
        'model':'gpt-4o-2024-11-20','input_price_per_million':2.5,'output_price_per_million':10.,
        'pricing_source':'https://developers.openai.com/api/docs/models/gpt-4o',
        'scope':'Independent relevance evaluation via OpenRouter GPT-4o, outside core rankers; no query or answer generation; token-based cost estimate without cached-input discount'}
    # Reservations and ledger writes happen synchronously before requests yield.
    semaphore=asyncio.Semaphore(8)
    async with httpx.AsyncClient(timeout=60) as client:
        for qid,q in qs.items():
            pool=fuse(q.candidates)
            for r in pool:r.passages=select(q.question,[r.abstract] if r.abstract else r.passages)
            docs={r.id:r for r in pool}
            wanted=sorted({d for r in rankings if r['query_id']==qid and r['success'] for d in r['result_ids']})
            async def evaluate_one(did):
                if (qid,did) in completed:return
                doc=docs[did];c=criteria[qid];record={'query_id':qid,'id':did}
                known=doc.identifiers.get('corpusid') in c['known_to_be_good']
                if known:
                    record.update(success=True,grade=3,known_positive_override=True,
                                  criterion_scores={v['name']:3 for v in c['relevance_criteria']})
                else:
                    prompt=judge_prompt(source,c['relevance_criteria'],doc)
                    # Exact tokenizer estimate + 200 framing-token allowance; fixed max output.
                    reserve=(len(encoder.encode(prompt))+200)*2.5/1e6+512*10/1e6
                    if spend['reserved_usd']+reserve>2.50:
                        raise ValueError('Independent judge reservation limit reached')
                    spend['reserved_usd']+=reserve;spend['requests']+=1
                    spend_path.write_text(json.dumps(spend,indent=2))
                    response=await client.post('https://openrouter.ai/api/v1/chat/completions',
                        headers={'Authorization':'Bearer '+openrouter_key},json={
                            'model':'openai/gpt-4o-2024-11-20','messages':[{'role':'user','content':prompt}],
                            'temperature':0,'max_tokens':512,'response_format':{'type':'json_object'}})
                    if not response.is_success:
                        record.update(success=False,http_status=response.status_code)
                    else:
                        payload=response.json();usage=payload['usage']
                        spend['prompt_tokens']+=usage['prompt_tokens'];spend['completion_tokens']+=usage['completion_tokens']
                        spend['estimated_actual_usd']+=usage['prompt_tokens']*2.5/1e6+usage['completion_tokens']*10/1e6
                        spend['reported_cost_usd']=spend.get('reported_cost_usd',0)+(usage.get('cost') or 0)
                        try:
                            record.update(parse_judgment(json.loads(payload['choices'][0]['message']['content']),c['relevance_criteria']),success=True,
                                          usage=usage,resolved_model=payload['model'])
                        except (ValueError,KeyError,TypeError):
                            record.update(success=False,error_type='InvalidJudgeOutput')
                    spend_path.write_text(json.dumps(spend,indent=2))
                with path.open('a') as f:f.write(json.dumps(record)+'\n')
                if not record['success']:
                    raise ValueError('Independent evaluator failed; no substitute judge or silent zero')
            async def limited(did):
                async with semaphore:
                    await evaluate_one(did)
            finished=await asyncio.gather(*(limited(did) for did in wanted),return_exceptions=True)
            if any(isinstance(v,BaseException) for v in finished):
                raise ValueError('One or more evaluator requests failed; successful judgments retained')
            print(json.dumps({'phase':'paperfinding_judge','query':qid,'judged_papers':len(wanted),
                              'estimated_actual_usd':spend['estimated_actual_usd']}),flush=True)


def report(out,normalizer):
    qs={q.id:q for q in load_snapshot(out/'frozen.jsonl')}
    criteria=json.loads((out/'criteria.json').read_text())
    rows=list({(r['query_id'],r['method']):r for r in
               [json.loads(s) for s in (out/'rankings.jsonl').read_text().splitlines()]}.values())
    judgments={(r['query_id'],r['id']):r for r in
               [json.loads(s) for s in (out/'judgments.jsonl').read_text().splitlines()] if r['success']}
    reference=json.loads(Path(normalizer).read_text());records=[]
    audits={a['query_id']:a['rows'] for a in
            [json.loads(s) for s in (out/'date-audit.jsonl').read_text().splitlines()]} if (out/'date-audit.jsonl').exists() else {}
    for row in rows:
        if not row['success']:continue
        qid=row['query_id'];ids=row['result_ids'];cr=criteria[qid]['relevance_criteria']
        qjudged={did:j for (qid2,did),j in judgments.items() if qid2==qid}
        if any(d not in qjudged for d in ids):raise ValueError('Unjudged final result')
        grades=[qjudged[d]['grade'] for d in ids]
        dcg=lambda gs:sum(g/math.log2(i+2) for i,g in enumerate(gs))
        ideal=dcg(sorted([j['grade'] for j in qjudged.values()],reverse=True)[:5])
        coverage=sum(c['weight']*max((qjudged[d]['criterion_scores'][c['name']] for d in ids),default=0)/3 for c in cr)
        pool=fuse(qs[qid].candidates)
        exa={r.id for r in pool if any(s.provider=='exa' for s in r.sources)}
        raw_exa=[r for r in audits.get(qid,[]) if r['provider']=='exa']
        exa.update(r['id'] for r in raw_exa)
        raw_exa_titles={title_key(r['title']) for r in raw_exa}
        novel={r.id for r in pool if r.id not in exa and title_key(r.title) not in raw_exa_titles}
        relevant=[d for d in ids if qjudged[d]['grade']>=2]
        perfect=sum(g==3 for g in grades)
        count=reference[qid];estimated=math.ceil(count*max(2,2/math.log(count) if count>1 else 10))
        recalled=sum(qjudged[d]['grade']==3 for d in ids[:estimated])/estimated
        hi,lo=dcg(sorted(grades,reverse=True)),dcg(sorted(grades))
        corrected=(dcg(grades)-lo)/(hi-lo) if hi!=lo else 0
        official_adapted=2*corrected*recalled/(corrected+recalled) if corrected and recalled else 0
        records.append({'query_id':qid,'method':row['method'],'nDCG@5_pooled':dcg(grades)/ideal if ideal else 0,
                        'P@5_grade2plus':len(relevant)/5,'perfect_count@5':perfect,
                        'weighted_aspect_coverage@5':coverage,'useful_non_exa_count@5':sum(d in novel for d in relevant),
                        'estimated_recall':recalled,'official_score_adapted':official_adapted,
                        'output_count':len(ids),'output_tokens':row['output_tokens']})
    common=set.intersection(*({r['query_id'] for r in records if r['method']==m} for m in METHODS))
    summary={};metric_names=['nDCG@5_pooled','P@5_grade2plus','weighted_aspect_coverage@5',
                             'useful_non_exa_count@5','official_score_adapted','output_tokens','output_count']
    for method in METHODS:
        selected=[r for r in records if r['method']==method and r['query_id'] in common]
        summary[method]={'queries':len(selected),**{k:mean(r[k] for r in selected) for k in metric_names}}
        if method!='single:exa':
            summary[method]['paired_vs_exa']=bootstrap_delta(
                {r['query_id']:r['P@5_grade2plus'] for r in selected},
                {r['query_id']:r['P@5_grade2plus'] for r in records if r['method']=='single:exa' and r['query_id'] in common})
    comparisons={}
    for left,right in [('jev','exa+jev'),('voyage','exa+voyage'),('jev','voyage'),('jev','decider')]:
        comparisons[left+'_vs_'+right]={metric:bootstrap_delta(
            {r['query_id']:r[metric] for r in records if r['method']==left and r['query_id'] in common},
            {r['query_id']:r[metric] for r in records if r['method']==right and r['query_id'] in common})
            for metric in ['P@5_grade2plus','weighted_aspect_coverage@5']}
    payload={'summary':summary,'records':records,'judged_pairs':len(judgments),'paired_queries':sorted(common),
             'pairwise':comparisons}
    (out/'results.json').write_text(json.dumps(payload,indent=2))
    lines=['# Sampled PaperFindingBench semantic evaluation','',
           f'{len(common)} sampled validation questions; independent GPT-4o rubric evaluation; five results / 1500 estimated tokens.',
           '','| Method | P@5 (grade >=2) | Pooled nDCG@5 | Aspect coverage | Useful non-Exa papers/query |',
           '|---|---:|---:|---:|---:|']
    for method,s in summary.items():
        lines.append(f"| {method} | {s['P@5_grade2plus']:.4f} | {s['nDCG@5_pooled']:.4f} | {s['weighted_aspect_coverage@5']:.4f} | {s['useful_non_exa_count@5']:.3f} |")
    lines+=['','## Scope and limitations','',
            '- This is a bounded-output adaptation, not an official AstaBench leaderboard result. Model judgments remain provisional without human adjudication.',
            '- Official per-question criteria, pinned evaluator model, weighted grading thresholds, and normalizer are used. Criterion names normalize case and underscores to avoid the reported scorer bug. Missing/invalid judgments stop evaluation.',
            '- Every unique final-result paper is judged once on the same 400-token source representation, irrespective of method. Provider/ranker identity and scores are hidden. Official known-positive overrides receive grade3. The rankers are not their own judges.',
            '- Pooled nDCG uses the union of judged final results as its ideal set; it is not exhaustive corpus recall. P@5 treats grades2 and3 as useful, and missing output slots as zero. Weighted aspect coverage is diagnostic, not proof that distinct papers add distinct findings.',
            '- Useful non-Exa papers means grade>=2 papers absent from all raw retrieved Exa results, including date-excluded hits, not merely absent from its final five. Exact normalized-title matches conservatively exclude possible same-paper versions from novelty credit.',
            '- All model blends rank the same provider-balanced shortlist. Exa+Jev and Exa+Voyage isolate reranking benefits from adding providers. Baselines share merged identities and text metadata.',
            '- Results lacking verifiable pre-June-2025 dates are excluded. This can disproportionately remove provider results with poor metadata; see date-audit.jsonl and per-provider accepted counts.',
            '- Semantic Scholar search failed with HTTP429 on 3/12 questions; metadata enrichment failed with HTTP429 on 4/12. A separately disclosed post-hoc metadata-success sensitivity analysis is in diagnostics.json; it does not replace this full-sample result.',
            '- Independent evaluator sees source passages/abstracts, not synthesized answers. Passage usefulness of the final shortened excerpt requires a separate assessment.',
            '- Source snapshots, full rankings, evaluator responses, query bootstrap intervals, output tokens, and separate spend ledgers are retained. Official adjusted-score adaptation is in JSON; its estimated recall and rank correction have different semantics from ordinary nDCG.']
    (out/'report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(summary,indent=2),flush=True)


def main():
    import getpass
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','collect-rank','judge','report'])
    parser.add_argument('--out',type=Path,default=Path('benchmarks/reports/scholar-benchmarks/paperfinding'))
    parser.add_argument('--raw',type=Path,default=Path('benchmarks/reports/scholar-benchmarks/raw'))
    args=parser.parse_args()
    if args.action=='prepare':prepare(args.raw/'paperfinding-validation_2025_05.json',args.out)
    elif args.action=='collect-rank':
        keys=json.loads(getpass.getpass('Provider/OpenRouter credentials JSON: '))
        asyncio.run(collect(args.out,keys));asyncio.run(rank(args.out,keys['openrouter']))
    elif args.action=='judge':
        key=getpass.getpass('OpenRouter evaluator key: ')
        asyncio.run(judge(args.out,key,args.raw/'paperfinding-relevance.py'))
    else:report(args.out,args.raw/'paperfinding-normalizer_reference.json')


if __name__=='__main__':main()
