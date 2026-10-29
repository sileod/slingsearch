"""Live provider ablation with fresh scholar questions and repeated web questions.

All paid calls share decision_budget.Transport's persistent session ceiling.
"""
import asyncio
from dataclasses import asdict
import getpass
import hashlib
import json
from pathlib import Path
import random
from statistics import mean
from time import perf_counter
import httpx
from benchmarks.datasets import Query,load_snapshot
from benchmarks.datasets.paperfinding import load as load_paperfinding
from benchmarks.decision_budget import Transport
from benchmarks.paperfinding_eval import metadata_id,judge,criterion_key
from benchmarks.evaluate import bootstrap_delta
from slingsearch import Config
from slingsearch.config import MODELS
from slingsearch.providers import PROVIDERS
from slingsearch.fusion import fuse,normalize,title_key
from slingsearch.diversity import balanced_shortlist
from slingsearch.passages import select,representation
from slingsearch.output import budget_results,evidence_tokens
from slingsearch.rankers import make_ranker

ROOT=Path('benchmarks/reports/decision-audit')


def prepare():
    manifest=json.loads((ROOT/'manifest.json').read_text())
    pf=load_paperfinding('benchmarks/reports/scholar-benchmarks/raw/paperfinding-validation_2025_05.json')
    chosen=[c for c in pf if c.query.id in manifest['fresh_paperfinding_query_ids']]
    old=load_snapshot('benchmarks/reports/anchor-holdout/frozen.jsonl')
    holdout=set(json.load(open('benchmarks/reports/anchor-holdout/results.json'))['results']['holdout']['query_ids'])
    web=sorted(random.Random(manifest['seed']).sample(sorted([q for q in old if q.id in holdout],key=lambda q:q.id),8),key=lambda q:q.id)
    manifest['web_query_ids']=[q.id for q in web]
    manifest['web_scope']='Previously seen benchmark questions, newly collected live outputs and independent relevance judgments. Not query-held-out validation.'
    for mode in ['scholar','web']:
        out=ROOT/mode;out.mkdir(exist_ok=True)
        if (out/'queries.jsonl').exists():continue
        if mode=='scholar':
            qs=[c.query for c in chosen];criteria={c.query.id:c.criteria for c in chosen}
        else:
            qs=[Query('web_'+q.id,q.question,'web',provenance=manifest['web_scope']) for q in web]
            criteria={q.id:{'known_to_be_good':[],'relevance_criteria':[
                {'name':'Useful evidence for the information need','weight':1,
                 'description':'The source must provide concrete, accurate-looking evidence that helps answer this complete information need, respecting its explicit constraints: '+q.question}]} for q in qs}
        (out/'queries.jsonl').write_text(''.join(json.dumps(asdict(q))+'\n' for q in qs))
        (out/'criteria.json').write_text(json.dumps(criteria,indent=2))
    (ROOT/'manifest.json').write_text(json.dumps(manifest,indent=2))


def names(mode):
    return ['exa','parallel','openalex','semantic_scholar'] if mode=='scholar' else ['exa','parallel','brave','tavily','firecrawl']


async def collect(mode,keys):
    out=ROOT/mode
    if (out/'frozen.jsonl').exists():return
    qs=load_snapshot(out/'queries.jsonl')
    previous={q.id:q for q in load_snapshot(out/'retrieval.jsonl')} if (out/'retrieval.jsonl').exists() else {}
    async with httpx.AsyncClient(transport=Transport(ROOT),timeout=60) as client:
        for index,q in enumerate(qs):
            if q.id in previous:qs[index]=previous[q.id];continue
            async def retrieve(name):
                adapter=PROVIDERS[name](client,keys[name],mode,year=(1900,2025) if mode=='scholar' else None)
                start=perf_counter()
                try:
                    docs=await adapter.search(q.question,10)
                    q.retrieval[name]={'success':True,'raw_count':len(docs),'reported_cost_usd':adapter.cost_usd}
                except Exception as error:
                    docs=[];q.retrieval[name]={'success':False,'error_type':type(error).__name__,
                        'http_status':getattr(getattr(error,'response',None),'status_code',None),'raw_count':0}
                q.retrieval[name]['latency_s']=perf_counter()-start
                return docs
            raw=[normalize(r) for batch in await asyncio.gather(*(retrieve(n) for n in names(mode))) for r in batch]
            # Save full retrieved evidence before any historical-date exclusion.
            with (out/'raw-retrieved.jsonl').open('a') as f:
                f.write(json.dumps({'query_id':q.id,'results':[asdict(r) for r in raw]})+'\n')
            date_by_id={}
            if mode=='scholar':
                lookups=sorted({metadata_id(r) for r in raw if metadata_id(r)})
                details={}
                if lookups:
                    for attempt in range(3):
                        await asyncio.sleep(3)
                        response=await client.post('https://api.semanticscholar.org/graph/v1/paper/batch',
                            headers={'x-api-key':keys['semantic_scholar']},
                            params={'fields':'title,year,publicationDate,abstract,externalIds,corpusId'},json={'ids':lookups})
                        if response.status_code!=429:break
                    q.retrieval['metadata']={'success':response.is_success,'http_status':response.status_code}
                    if response.is_success:details={pid:p for pid,p in zip(lookups,response.json()) if p}
                for r in raw:
                    paper=details.get(metadata_id(r))
                    if paper:
                        r.year=paper.get('year') or r.year;r.abstract=paper.get('abstract') or r.abstract
                        r.identifiers.update({k.lower():str(v) for k,v in (paper.get('externalIds') or {}).items() if v})
                        r.identifiers['corpusid']=str(paper['corpusId']);r.identifiers['semantic_scholar']=paper['paperId']
                        r.doi=r.doi or r.identifiers.get('doi');normalize(r)
                        if paper.get('publicationDate'):date_by_id[r.id]=paper['publicationDate']
            pool=fuse(raw)
            audit=[];q.candidates=[]
            for r in pool:
                date=date_by_id.get(r.id)
                valid=mode=='web' or (date<'2025-06-01' if date else r.year is not None and r.year<2025)
                audit.append({'id':r.id,'title':r.title,'year':r.year,'date':date,'accepted':valid,
                              'sources':[asdict(s) for s in r.sources]})
                if valid:q.candidates.append(r)
            with (out/'date-audit.jsonl').open('a') as f:f.write(json.dumps({'query_id':q.id,'rows':audit})+'\n')
            with (out/'retrieval.jsonl').open('a') as f:f.write(json.dumps(asdict(q))+'\n')
            print(json.dumps({'phase':'retrieval','mode':mode,'query':q.id,'unique_accepted':len(q.candidates),
                              'status':{n:q.retrieval[n]['success'] for n in names(mode)}}),flush=True)
            await asyncio.sleep(2)
    (out/'frozen.jsonl').write_text(''.join(json.dumps(asdict(q))+'\n' for q in qs))
    (out/'manifest.json').write_text(json.dumps({'snapshot_sha256':hashlib.sha256((out/'frozen.jsonl').read_bytes()).hexdigest(),
        'criteria_sha256':hashlib.sha256((out/'criteria.json').read_bytes()).hexdigest()},indent=2))


def methods(mode):
    return ['single:'+n for n in names(mode)]+['rrf','jev_single','jev_packed5','voyage','exa+voyage','parallel+voyage']


async def rank(mode,key):
    out=ROOT/mode;path=out/'rankings.jsonl'
    previous={(r['query_id'],r['method']) for r in [json.loads(s) for s in path.read_text().splitlines()] if r['success']} if path.exists() else set()
    async with httpx.AsyncClient(transport=Transport(ROOT),timeout=60) as client:
        for q in load_snapshot(out/'frozen.jsonl'):
            pool=fuse(q.candidates)
            for r in pool:r.passages=select(q.question,[r.abstract] if r.abstract else r.passages)
            shortlist=balanced_shortlist(pool,20)
            singles={p:sorted([r for r in pool if any(s.provider==p for s in r.sources)],
                            key=lambda r:min(s.rank for s in r.sources if s.provider==p)) for p in names(mode)}
            for method in methods(mode):
                if (q.id,method) in previous:continue
                record={'query_id':q.id,'method':method};start=perf_counter()
                try:
                    if method.startswith('single:'):ranked=singles[method.split(':')[1]]
                    elif method=='rrf':ranked=shortlist
                    else:
                        candidates=singles[method.split('+')[0]] if '+' in method else shortlist
                        backend='jev' if method.startswith('jev') else 'voyage'
                        config=Config(api_keys={'openrouter':key},candidate_tokens=400,ranking_concurrency=12,
                                      decision_batch_size=5 if method=='jev_packed5' else 1)
                        ranker=make_ranker(backend,client,config)
                        ranked=await ranker.rank(q.question,candidates) if candidates else []
                        record.update(usage=ranker.usage,reported_cost_usd=ranker.cost_usd)
                    output=budget_results(ranked,5,1500)
                    record.update(success=True,result_ids=[r.id for r in output],output_tokens=evidence_tokens(output),
                                  output=[asdict(r) for r in output])
                except Exception as error:record.update(success=False,error_type=type(error).__name__,
                    http_status=getattr(getattr(error,'response',None),'status_code',None))
                record['latency_s']=perf_counter()-start
                with path.open('a') as f:f.write(json.dumps(record)+'\n')
                print(json.dumps({'phase':'rank','mode':mode,'query':q.id,'method':method,'success':record['success']}),flush=True)


def report(mode):
    out=ROOT/mode;qs={q.id:q for q in load_snapshot(out/'frozen.jsonl')}
    criteria=json.load(open(out/'criteria.json'))
    rows=list({(r['query_id'],r['method']):r for r in [json.loads(s) for s in (out/'rankings.jsonl').read_text().splitlines()]}.values())
    judgments={(r['query_id'],r['id']):r for r in [json.loads(s) for s in (out/'judgments.jsonl').read_text().splitlines()] if r['success']}
    raw={r['query_id']:r['results'] for r in [json.loads(s) for s in (out/'raw-retrieved.jsonl').read_text().splitlines()]}
    records=[]
    for row in rows:
        if not row['success']:continue
        qid=row['query_id'];ids=row['result_ids']
        if any((qid,d) not in judgments for d in ids):raise ValueError('Incomplete final judgments')
        grades=[judgments[qid,d]['grade'] for d in ids]
        coverage=sum(c['weight']*max((judgments[qid,d]['criterion_scores'][c['name']] for d in ids),default=0)/3 for c in criteria[qid]['relevance_criteria'])
        raw_exa=[r for r in raw[qid] if any(s['provider']=='exa' for s in r['sources'])]
        exa_titles={title_key(r['title']) for r in raw_exa};exa_ids={r['id'] for r in raw_exa}
        docs={r.id:r for r in fuse(qs[qid].candidates)}
        novel=sum(judgments[qid,d]['grade']>=2 and d not in exa_ids and title_key(docs[d].title) not in exa_titles
                  and not any(s.provider=='exa' for s in docs[d].sources) for d in ids)
        records.append({'query_id':qid,'method':row['method'],'P@5':sum(g>=2 for g in grades)/5,
                        'aspect_coverage':coverage,'useful_non_exa':novel,'output_tokens':row['output_tokens']})
    common=set.intersection(*({r['query_id'] for r in records if r['method']==m} for m in methods(mode)))
    summaries={m:{'queries':len(common),**{key:mean(r[key] for r in records if r['method']==m and r['query_id'] in common)
                                        for key in ['P@5','aspect_coverage','useful_non_exa','output_tokens']}} for m in methods(mode)}
    best=max(['single:'+n for n in names(mode)],key=lambda m:summaries[m]['P@5'])
    pairs=[('rrf',best),('jev_single',best),('jev_packed5',best),('voyage',best),
           ('jev_packed5','jev_single'),('voyage','exa+voyage'),('voyage','parallel+voyage')]
    compared={left+'_vs_'+right:bootstrap_delta(
        {r['query_id']:r['P@5'] for r in records if r['method']==left and r['query_id'] in common},
        {r['query_id']:r['P@5'] for r in records if r['method']==right and r['query_id'] in common}) for left,right in pairs}
    payload={'mode':mode,'summary':summaries,'best_single_selected_on_sample':best,'comparisons':compared,
             'records':records,'paired_queries':sorted(common),'judgment_pairs':len(judgments)}
    (out/'results.json').write_text(json.dumps(payload,indent=2));print(json.dumps(summaries,indent=2),flush=True)


async def run(keys):
    for mode in ['scholar','web']:
        await collect(mode,keys);await rank(mode,keys['openrouter'])
        await judge(ROOT/mode,keys['openrouter'],
                    Path('benchmarks/reports/scholar-benchmarks/raw/paperfinding-relevance.py'),transport=Transport(ROOT))
        report(mode)


if __name__=='__main__':
    import sys
    if sys.argv[1]=='prepare':prepare()
    elif sys.argv[1]=='report':
        for mode in ['scholar','web']:report(mode)
    else:asyncio.run(run(json.loads(getpass.getpass('Provider/OpenRouter credentials JSON: '))))
