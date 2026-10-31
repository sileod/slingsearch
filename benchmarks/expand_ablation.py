"""Paired Expand ablation on newly sampled FreshQA and SAGE questions.

Run `prepare`, `collect`, `rrf`, `free`, `blind`, then expand_judge and expand_report. Retrieval is frozen once,
shared across all methods. Model failures are failed measurements, not fallbacks.
All paid requests use the persistent session budget. Raw datasets stay private.
"""
import argparse
import asyncio
from copy import deepcopy
from dataclasses import asdict
import csv
import hashlib
import json
from pathlib import Path
import random
from statistics import mean,median
from time import perf_counter
import re
import httpx
from benchmarks.datasets import Query,load_snapshot
from benchmarks.datasets import sage
from benchmarks.decision_budget import Transport
from benchmarks.academic_eval import gold_aliases,score
from benchmarks.evaluate import bootstrap_delta,metrics
from slingsearch import Config,Result
from slingsearch.fusion import fuse,normalize
from slingsearch.diversity import reserve_shortlist
from slingsearch.passages import prepare as prepare_passages
from slingsearch.output import budget_results,evidence_tokens
from slingsearch.providers import PROVIDERS
from slingsearch.providers.semantic_scholar import SemanticScholar
from slingsearch.rankers import make_ranker

ROOT=Path('benchmarks/reports/expand-ablation')
METHODS=('default','default+expand','expand')
SEED=20261031


def dump(path,value):path.write_text(json.dumps(value,indent=2))
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare():
    if (ROOT/'queries.jsonl').exists():raise ValueError('Prepared sample already exists; preserve it')
    ROOT.mkdir(exist_ok=True,parents=True)
    seen=set()
    for name in ('frozen.jsonl','sample.json','queries.jsonl','retrieval.jsonl'):
        for path in Path('benchmarks/reports').rglob(name):
            if ROOT in path.parents:continue
            try:seen.update(q.question.strip() for q in load_snapshot(path))
            except (ValueError,KeyError,TypeError):pass
    rows=list(csv.DictReader((ROOT/'freshqa.private.csv').open()))
    rng=random.Random(SEED);selected=[]
    for hops in ('one-hop','multi-hop'):
        pool=[r for r in rows if r['fact_type']=='never-changing' and r['false_premise']=='FALSE'
              and r['num_hops']==hops and r['question'].strip() not in seen and 'covid' not in r['question'].lower()]
        chosen=rng.sample(sorted(pool,key=lambda r:r['question']),4)
        for row in chosen:
            identifier=next(v for k,v in row.items() if k.endswith(' id'))
            selected.append(Query('freshqa_'+identifier,row['question'],'web',provenance='FreshQA April 21 2026 stable true-premise subset; held out from prior local experiments',
                retrieval={'reference_answers':[row['answer_'+str(i)] for i in range(10) if row['answer_'+str(i)]],
                           'reference_urls':row['source'].splitlines(),'hops':hops}))
    for kind in sage.KINDS:
        for domain in sage.DOMAINS:
            path=Path('benchmarks/reports/academic-samples/raw')/f'sage_{kind}_{domain}.json'
            pool=[q for q in sage.load(path,kind=kind,domain=domain) if q.question.strip() not in seen and 'covid' not in q.question.lower()]
            selected.append(rng.choice(sorted(pool,key=lambda q:q.id)))
    assert not any(q.question.strip() in seen for q in selected)
    (ROOT/'queries.jsonl').write_text(''.join(json.dumps(asdict(q))+'\n' for q in selected))
    manifest={'seed':SEED,'selected_ids':[q.id for q in selected],'counts':{'web':8,'scholar':8},
        'query_sha256':digest(ROOT/'queries.jsonl'),'freshqa_sha256':digest(ROOT/'freshqa.private.csv'),
        'freshqa_source':'https://github.com/freshllms/freshqa','freshqa_sheet':'1_8mi-yuK30mvoDJu1KQXD6ODem7MKMcIgVAwDSzJkjM',
        'prior_questions_excluded':len(seen),'sources':'FreshQA stable true-premise questions and official SAGE, one fresh question per task/domain stratum',
        'per_provider':10,'shortlist':20,'candidate_tokens':400,'max_results':5,'output_tokens':1500,'min_relevance':.5,
        'methods':list(METHODS),'defaults':{'web':list(Config().web_providers),'scholar':list(Config().scholar_providers)},
        'weights':Config().scholar_provider_weights,'model_requested':'~typesafe/jev-latest',
        'precommitted_policy':'Paired add-one-provider ablation. No provider weight or cutoff tuning. Missing model calls are not successful default evaluations.',
        'rrf_policy':'Free supplemental fusion comparison without a relevance cutoff; explicitly not the shipped model default',
        'source_text_policy':'Identical passage selection and budgets for each arm; no provider scores or synthesized answers used'}
    dump(ROOT/'manifest.json',manifest)
    print(json.dumps(manifest,indent=2))


async def collect():
    config=Config();queries=load_snapshot(ROOT/'queries.jsonl')
    completed={q.id for q in load_snapshot(ROOT/'frozen.jsonl')} if (ROOT/'frozen.jsonl').exists() else set()
    gold_path=ROOT/'gold-metadata.private.json';gold=json.loads(gold_path.read_text()) if gold_path.exists() else {}
    async with httpx.AsyncClient(transport=Transport('benchmarks/reports/decision-audit'),timeout=35) as client:
        s2=SemanticScholar(client,config.api_keys.get('semantic_scholar',''),'scholar')
        for q in queries:
            if q.id in completed:continue
            names=[*getattr(config,q.mode+'_providers'),'expand']
            async def retrieve(name):
                a=PROVIDERS[name](client,config.api_keys.get(name,''),q.mode);start=perf_counter()
                try:
                    docs=await a.search(q.question,10);status={'success':True,'count':len(docs),'reported_cost_usd':a.cost_usd}
                except Exception as error:
                    docs=[];status={'success':False,'error_type':type(error).__name__,'http_status':getattr(getattr(error,'response',None),'status_code',None)}
                status['seconds']=perf_counter()-start;q.retrieval[name]=status;return docs
            batches=await asyncio.gather(*(retrieve(name) for name in names));raw=[normalize(r) for b in batches for r in b]
            # Resolve identities for scoring only. Metadata never changes
            # retrieved passages, provider ranks or the production candidate pool.
            if q.mode=='scholar':
                from benchmarks.academic_eval import lookup_id,aliases
                lookup={lookup_id(r) for r in raw if lookup_id(r)}
                lookup.update(pid.removeprefix('semantic_scholar:') for pid in q.qrels)
                missing=sorted(pid for pid in lookup if pid not in gold)
                for start in range(0,len(missing),100):
                    batch=missing[start:start+100]
                    try:
                        papers=await s2.request('POST','https://api.semanticscholar.org/graph/v1/paper/batch',
                            headers={'x-api-key':s2.key},params={'fields':'title,year,abstract,externalIds,corpusId'},json={'ids':batch})
                        gold.update({pid:p for pid,p in zip(batch,papers) if p})
                    except Exception as error:q.retrieval['metadata']={'success':False,'error_type':type(error).__name__}
                dump(gold_path,gold)
            q.candidates=raw
            with (ROOT/'frozen.jsonl').open('a') as f:f.write(json.dumps(asdict(q))+'\n')
            print(json.dumps({'query':q.id,'mode':q.mode,'successful':[n for n in names if q.retrieval[n]['success']]}),flush=True)
    manifest=json.loads((ROOT/'manifest.json').read_text());manifest['frozen_sha256']=digest(ROOT/'frozen.jsonl');dump(ROOT/'manifest.json',manifest)


def candidates_for(q,method):
    config=Config();selected={'expand'} if method=='expand' else set(getattr(config,q.mode+'_providers')) | ({'expand'} if method.endswith('+expand') else set())
    raw=[]
    for result in deepcopy(q.candidates):
        result.sources=[s for s in result.sources if s.provider in selected]
        if result.sources:raw.append(result)
    weights=config.scholar_provider_weights if q.mode=='scholar' else {}
    fused=fuse(raw,weights)
    shortlist=reserve_shortlist(fused,20,weights)
    for r in shortlist:prepare_passages(r,q.question)
    return fused,shortlist


async def rank(model=False):
    config=Config();path=ROOT/('model-rankings.private.jsonl' if model else 'rrf-rankings.private.jsonl')
    done={(r['query_id'],r['method']) for r in [json.loads(s) for s in path.read_text().splitlines()] if r['success']} if path.exists() else set()
    async with httpx.AsyncClient(transport=Transport('benchmarks/reports/decision-audit'),timeout=35) as client:
        for q in load_snapshot(ROOT/'frozen.jsonl'):
            for method in METHODS:
                if (q.id,method) in done:continue
                fused,candidates=candidates_for(q,method);row={'query_id':q.id,'method':method,'backend':'jev' if model else 'rrf'}
                start=perf_counter()
                try:
                    if model:
                        backend=make_ranker('jev',client,config);candidates=await backend.rank(q.question,candidates)
                        row.update(model=','.join(sorted(backend.resolved_models)),usage=backend.usage,reported_cost_usd=backend.cost_usd)
                        candidates=[r for r in candidates if (r.model_score-1)/4>=.5]
                    output=budget_results(candidates,5,1500)
                    row.update(success=True,output=[asdict(r) for r in output],result_ids=[r.id for r in output],
                        shortlist_ids=[r.id for r in candidates],pool_ids=[r.id for r in fused],output_tokens=evidence_tokens(output))
                except Exception as error:row.update(success=False,error_type=type(error).__name__)
                row['ranking_seconds']=perf_counter()-start
                with path.open('a') as f:f.write(json.dumps(row)+'\n')
    print('Ranking snapshot saved',path)


def evidence_key(qid, paper):
    return hashlib.sha256(json.dumps([qid,paper['id'],paper['title'],paper['passages']],sort_keys=True).encode()).hexdigest()


async def free_rank():
    from benchmarks.span_free import score_documents,key,MODEL
    config=Config();path=ROOT/'free-rankings.private.jsonl'
    completed={(r['query_id'],r['method']) for r in [json.loads(s) for s in path.read_text().splitlines()] if r['success']} if path.exists() else set()
    async with httpx.AsyncClient(transport=Transport('benchmarks/reports/decision-audit'),timeout=45) as client:
        for q in load_snapshot(ROOT/'frozen.jsonl'):
            if all((q.id,m) in completed for m in METHODS):continue
            arms={m:candidates_for(q,m) for m in METHODS};union=[r for _,pool in arms.values() for r in pool]
            started=perf_counter()
            try:
                scored,usage=await score_documents(client,config.api_keys['openrouter'],q.question,union,ROOT/'free-score-cache.private.json')
                for method,(pool,candidates) in arms.items():
                    for r in candidates:r.relevance_score=scored[key(q.question,r)]['score'];r.model_score=r.relevance_score
                    ranked=sorted(candidates,key=lambda r:(-r.relevance_score,-r.rrf_score,r.id))
                    ranked=[r for r in ranked if r.relevance_score>=.5]
                    output=budget_results(ranked,5,1500)
                    row={'query_id':q.id,'method':method,'backend':'span_free','success':True,
                        'requested_model':MODEL,'resolved_models':sorted({v['model'] for v in scored.values()}),
                        'output':[asdict(r) for r in output],'result_ids':[r.id for r in output],
                        'shortlist_ids':[r.id for r in ranked],'pool_ids':[r.id for r in pool],
                        'output_tokens':evidence_tokens(output),'shared_scoring_seconds':perf_counter()-started,
                        'shared_scoring_usage':usage}
                    with path.open('a') as f:f.write(json.dumps(row)+'\n')
                print(json.dumps({'query':q.id,'backend':'span_free','qualified':{m:sum(scored[key(q.question,r)]['score']>=.5 for r in c) for m,(_,c) in arms.items()}}),flush=True)
            except Exception as error:
                print(json.dumps({'query':q.id,'status':'failed','error_type':type(error).__name__}),flush=True)
                # Keep successful per-document scores for one bounded resume.
                for method in METHODS:
                    with path.open('a') as f:f.write(json.dumps({'query_id':q.id,'method':method,'backend':'span_free','success':False,'error_type':type(error).__name__})+'\n')


def blinded():
    qs={q.id:q for q in load_snapshot(ROOT/'frozen.jsonl')};docs={}
    for path in (ROOT/'rrf-rankings.private.jsonl',ROOT/'model-rankings.private.jsonl',ROOT/'free-rankings.private.jsonl'):
        if not path.exists():continue
        for line in path.read_text().splitlines():
            row=json.loads(line)
            if not row['success'] or qs[row['query_id']].mode!='web':continue
            for paper in row['output']:
                key=(row['query_id'],paper['id'],evidence_key(row['query_id'],paper));docs.setdefault(key,paper)
    shuffled=list(docs.items());random.Random(SEED).shuffle(shuffled);view=[];mapping={}
    for i,((qid,pid,text_hash),paper) in enumerate(shuffled):
        identifier=f'D{i:03d}';mapping[identifier]={'query_id':qid,'id':pid,'evidence_key':text_hash}
        view.append({'id':identifier,'question':qs[qid].question,'reference_answers':qs[qid].retrieval['reference_answers'],
            'title':paper['title'],'url':paper['url'],'passages':paper['passages']})
    dump(ROOT/'blinded.private.json',view);dump(ROOT/'blind-map.private.json',mapping)
    print('Blinded web excerpts',len(view))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','collect','rrf','model','free','blind']);args=parser.parse_args()
    if args.action=='prepare':prepare()
    elif args.action=='blind':blinded()
    elif args.action=='free':asyncio.run(free_rank())
    else:asyncio.run(collect() if args.action=='collect' else rank(args.action=='model'))
