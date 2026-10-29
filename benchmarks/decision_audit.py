"""Fresh sampled decision packing and evidence-check ablation, with one global budget."""
import asyncio
from dataclasses import asdict
import getpass
import hashlib
import json
from pathlib import Path
from statistics import mean,median
from time import perf_counter
import httpx
from benchmarks.datasets import doris
from benchmarks.decision_budget import Transport
from benchmarks.scholar_eval import lexical_order,evaluate
from benchmarks.evaluate import bootstrap_delta
from slingsearch import Config
from slingsearch.rankers import make_ranker

ROOT=Path('benchmarks/reports/decision-audit')


def prepare():
    manifest=json.loads((ROOT/'manifest.json').read_text())
    cases=doris.load('benchmarks/reports/scholar-benchmarks/raw/doris-native.json',manifest['doris_indices'])
    for case in cases:case.query.candidates=lexical_order(case.query.question,case.query.candidates)[:20]
    doris.save_cases(cases,ROOT/'doris-frozen.json')
    manifest['doris_sha256']=hashlib.sha256((ROOT/'doris-frozen.json').read_bytes()).hexdigest()
    (ROOT/'manifest.json').write_text(json.dumps(manifest,indent=2))


async def run(key):
    cases=doris.load_cases(ROOT/'doris-frozen.json');path=ROOT/'doris-rankings.jsonl'
    records=[json.loads(s) for s in path.read_text().splitlines()] if path.exists() else []
    complete={(r['query_id'],r['method']) for r in records if r['success']}
    async with httpx.AsyncClient(transport=Transport(ROOT),timeout=60) as client:
        for case in cases:
            for method in ['jev_single','jev_packed5','jev_three_judgments','voyage']:
                if (case.query.id,method) in complete:continue
                config=Config(api_keys={'openrouter':key},candidate_tokens=400,ranking_concurrency=12,
                              decision_batch_size=5 if method=='jev_packed5' else 1,
                              decision_evidence_checks=method=='jev_three_judgments')
                ranker=make_ranker('voyage' if method=='voyage' else 'jev',client,config)
                start=perf_counter();record={'query_id':case.query.id,'method':method}
                try:
                    ranked=await ranker.rank(case.query.question,case.query.candidates)
                    record.update(success=True,result_ids=[r.id for r in ranked],
                                  metrics=evaluate(case,[r.id for r in ranked]),usage=ranker.usage,
                                  reported_cost_usd=ranker.cost_usd,resolved_models=sorted(ranker.resolved_models),
                                  evidence=ranker.judgments if method!='voyage' else {})
                except Exception as error:record.update(success=False,error_type=type(error).__name__,
                    http_status=getattr(getattr(error,'response',None),'status_code',None))
                record['latency_s']=perf_counter()-start
                with path.open('a') as f:f.write(json.dumps(record)+'\n')
                print(json.dumps({'query':case.query.id,'method':method,'success':record['success'],
                                  'latency_s':round(record['latency_s'],2)}),flush=True)
    report()


def report():
    rows=list({(r['query_id'],r['method']):r for r in
               [json.loads(s) for s in (ROOT/'doris-rankings.jsonl').read_text().splitlines()]}.values())
    methods=['jev_single','jev_packed5','jev_three_judgments','voyage']
    common=set.intersection(*({r['query_id'] for r in rows if r['method']==m and r['success']} for m in methods))
    summaries={}
    for method in methods:
        selected=[r for r in rows if r['method']==method and r['query_id'] in common and r['success']]
        summaries[method]={'queries':len(selected),'nDCG@5':mean(r['metrics']['nDCG@5'] for r in selected),
                           'latency_p50_s':median(r['latency_s'] for r in selected),
                           'mean_reported_cost_usd':mean(r.get('reported_cost_usd') or 0 for r in selected),
                           'mean_input_tokens':mean(r['usage'].get('input_tokens',r['usage'].get('total_tokens',0)) for r in selected),
                           'mean_requests':mean(r['usage'].get('ranking_requests',1) for r in selected),
                           'cache_telemetry_reported':any('cached_tokens' in r['usage'] for r in selected)}
    paired={m:bootstrap_delta({r['query_id']:r['metrics']['nDCG@5'] for r in rows if r['method']==m and r['query_id'] in common},
                             {r['query_id']:r['metrics']['nDCG@5'] for r in rows if r['method']=='jev_single' and r['query_id'] in common})
            for m in methods if m!='jev_single'}
    (ROOT/'doris-summary.json').write_text(json.dumps({'summary':summaries,'paired_vs_single':paired},indent=2))
    print(json.dumps(summaries,indent=2),flush=True)


if __name__=='__main__':
    import sys
    if sys.argv[1]=='prepare':prepare()
    elif sys.argv[1]=='report':report()
    else:asyncio.run(run(getpass.getpass('OpenRouter key: ')))
