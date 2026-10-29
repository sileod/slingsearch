"""Explicit paid sample run; credentials are never written to artifacts.

The restricted transport reserves <= $3.50 for ranking, based on at least
$0.001/request or ten times the verified highest input rate per request byte.
At most 1,250 requests of <= 30 KB are sent. Exa/OpenAlex retrieval costs are
separate: defaults use only 8 queries x 10 results/provider. No retries.
"""
import argparse,asyncio,copy,hashlib,json,random,sys
from collections import defaultdict
from dataclasses import asdict
from decimal import Decimal
from pathlib import Path
from statistics import mean
import httpx
from slingsearch import Config
from slingsearch.config import MODELS
from benchmarks.sample import web_sample
from benchmarks.datasets import Query,rows
from benchmarks.live import snapshot
from benchmarks.compare import compare,report,paired,operational
from benchmarks.evaluate import bootstrap_delta
from slingsearch.fusion import normalize_doi
parser=argparse.ArgumentParser(description='Run a small reproducible retrieval/ranking comparison with bounded ranking spend')
parser.add_argument('--web-results', required=True)
parser.add_argument('--web-judgments', required=True)
parser.add_argument('--web-queries', required=True)
parser.add_argument('--web-runs', required=True)
parser.add_argument('--litsearch-queries', required=True, help='Native query JSON rows or HF first-rows JSON')
parser.add_argument('--output', required=True)
parser.add_argument('--stdin-keys', action='store_true', help='Read one JSON dictionary of credentials from stdin; otherwise use environment')
parser.add_argument('--web-size', type=int, default=20)
parser.add_argument('--scholar-size', type=int, default=8)
args=parser.parse_args()
keys=json.loads(sys.stdin.readline()) if args.stdin_keys else Config().api_keys
out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
class Guard(httpx.AsyncBaseTransport):
 def __init__(self):
  self.inner=httpx.AsyncHTTPTransport(retries=0);self.reserved=Decimal(0);self.reported=Decimal(0);self.calls=0
 async def handle_async_request(self,request):
  body=json.loads(request.content)
  if request.url.host!='openrouter.ai' or body.get('model') not in MODELS.values() or request.url.path not in ['/api/alpha/decisions','/api/v1/rerank']:raise ValueError('Outside ranking scope')
  if len(request.content)>30000:raise ValueError('Oversized request')
  amount=max(Decimal('.001'),Decimal(len(request.content))*Decimal('.000000042')*10)
  if self.reserved+amount>Decimal('3.5') or self.calls>=1250:raise ValueError('Budget reservation ceiling')
  self.reserved+=amount;self.calls+=1
  response=await self.inner.handle_async_request(request);await response.aread()
  try:
   usage=response.json().get('usage') or {}
   if usage.get('cost') is not None:self.reported+=Decimal(str(usage['cost']))
  except Exception:pass
  return response
 async def aclose(self):pass

async def main():
 web,manifest=web_sample(args.web_results,args.web_judgments,args.web_queries,args.web_runs,size=args.web_size)
 raw=json.load(open(args.litsearch_queries));data=raw['rows'] if isinstance(raw,dict) else [{'row':r} for r in raw];indices=sorted(random.Random(42).sample(range(len(data)),min(args.scholar_size,len(data))))
 scholar=[Query('litsearch-'+str(i),data[i]['row']['query'],'scholar',qrels={'corpusid:'+str(pid):1 for pid in data[i]['row']['corpusids']},provenance='LitSearch seed42 sample of provided questions; live OpenAlex/Semantic Scholar/Exa retrieval; DOI and exact scholarly identifier matching only') for i in indices]
 pids=sorted({pid for i in indices for pid in data[i]['row']['corpusids']})
 async with httpx.AsyncClient(timeout=30) as client:
  resp=await client.post('https://api.semanticscholar.org/graph/v1/paper/batch',headers={'x-api-key':keys['semantic_scholar']},params={'fields':'title,year,externalIds,corpusId'},json={'ids':['CorpusId:'+str(pid) for pid in pids]})
  resp.raise_for_status();papers=resp.json()
 corpus={}
 for p in papers:
  if p:
   ids={k.lower():str(v) for k,v in (p.get('externalIds') or {}).items() if v}
   ids['corpusid']=str(p['corpusId']);ids['semantic_scholar']=p['paperId']
   corpus['corpusid:'+str(p['corpusId'])]={'title':p['title'],'year':p.get('year'),'doi':ids.get('doi'),'identifiers':ids}
 (out/'gold-identifiers.json').write_text(json.dumps(corpus,indent=2))
 config=Config(api_keys=keys,timeout=30,ranking_concurrency=5,candidate_tokens=300)
 # Titles alone never assign relevance in this run: only DOI or exact identifier.
 exact_corpus={k:{**p,'title':'','year':None} for k,p in corpus.items()}
 for i,q in enumerate(scholar):
  await snapshot([q],['openalex','semantic_scholar','exa'],config=config,limit=10,corpus=exact_corpus)
  print(json.dumps({'phase':'scholar_retrieval','query':i+1,'results':len(q.candidates),'status':{p:r['success'] for p,r in q.retrieval.items()}}),flush=True)
 queries=web+scholar
 (out/'frozen.jsonl').write_text(''.join(json.dumps(asdict(q))+'\n' for q in queries))
 manifest.update({'scholar_indices':indices,'scholar_sampling_frame':len(data),'shortlist':20,'candidate_tokens':300,'web_qrel_source':'Released auto_heuristic_v1; not independent human labels','scholar_matching':'Exact DOI/identifier only; relevant papers not injected','prior_reported_ranking_cost_usd':0})
 (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
 guard=Guard();records=[]
 for i,q in enumerate(queries):
  if not q.candidates:
   print(json.dumps({'phase':'ranking','query':q.id,'error':'no candidates'}),flush=True);continue
  result=await compare([q],config=config,shortlist=20,transport=guard)
  records.extend(result)
  checkpoint={'records':records,'paired':paired(records),'operational':operational(records),'ranking_cost_usd':float(guard.reported),'reserved_usd':float(guard.reserved),'requests':guard.calls}
  (out/'results.json').write_text(json.dumps(checkpoint,indent=2))
  print(json.dumps({'phase':'ranking','query':i+1,'of':len(queries),'reported_cost':float(guard.reported),'reserved':float(guard.reserved),'failures':[r['method'] for r in result if not r['success']]}),flush=True)
 await guard.inner.aclose()
 # Extra no-cost RRF experiment restricted to default web trio.
 trio=[]
 for q in web:
  v=copy.deepcopy(q);v.candidates=[r for r in v.candidates if r.sources[0].provider in {'exa','parallel','brave'}];v.retrieval={p:r for p,r in v.retrieval.items() if p in {'exa','parallel','brave'}};trio.append(v)
 trio_records=await compare(trio,methods=['rrf'],config=Config(api_keys={}),shortlist=20)
 (out/'default-web-trio.json').write_text(json.dumps(trio_records,indent=2))
 summary={}
 for mode in ['web','scholar']:
  rs=[r for r in records if r['mode']==mode]
  methods=sorted({r['method'] for r in rs})
  summary[mode]={m:{metric:mean(r['metrics'][metric] for r in rs if r['method']==m and metric in r['metrics']) for metric in ['nDCG@5','nDCG@10','R@10','R@20','RR@10'] if any(metric in r['metrics'] for r in rs if r['method']==m)} for m in methods}
  singles=[m for m in methods if m.startswith('single:')]
  best=max(singles,key=lambda m:summary[mode][m].get('nDCG@5',0))
  values=lambda m:{r['query_id']:r['metrics']['nDCG@5'] for r in rs if r['method']==m and 'nDCG@5' in r['metrics']}
  summary[mode]['rrf_vs_best_single']=bootstrap_delta(values('rrf'),values(best))
  summary[mode]['best_single']=best
 (out/'summary.json').write_text(json.dumps(summary,indent=2))
 retrieval_cost=sum(r.get('cost') or 0 for q in scholar for r in q.retrieval.values())
 (out/'report.md').write_text('# Sampled method comparison\n\nSampled frozen web queries with heuristic labels and live LitSearch questions. Seed 42. Scholar matching uses DOI/identifiers only; no relevant papers injected. These samples give preliminary evidence.\n\n'+report(records)+f'\nRanking spend: ${guard.reported}; known retrieval spend: ${retrieval_cost}. Total known reported spend: ${float(guard.reported)+retrieval_cost:.8f}.\n')
 print('SUMMARY '+json.dumps(summary),flush=True)
 print(json.dumps({'reported_ranking':float(guard.reported),'reported_retrieval':retrieval_cost,'total_known':float(guard.reported)+retrieval_cost,'reserved':float(guard.reserved),'calls':guard.calls}),flush=True)
asyncio.run(main())
