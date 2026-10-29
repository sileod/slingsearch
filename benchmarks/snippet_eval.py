"""Small live S2 paper/snippet and frozen-shortlist enrichment pilot. No model calls.

python -m benchmarks.snippet_eval
Private academic snapshots and retrieval credentials are required. Saves raw
responses locally; publishes aggregates only. Six reused queries are exploratory.
"""
import asyncio
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
from statistics import mean
from time import perf_counter
import httpx
from slingsearch import Config
from slingsearch.fusion import fuse
from slingsearch.output import budget_results, evidence_tokens
from slingsearch.passages import prepare
from slingsearch.providers.semantic_scholar import SemanticScholar
from slingsearch.snippets import enrich
from benchmarks.academic_eval import score
from benchmarks.datasets import load_snapshot
from benchmarks.decision_budget import Transport

ROOT=Path('benchmarks/reports/snippet-pilot')


async def run():
    source=Path('benchmarks/reports/academic-sources')
    queries=load_snapshot(source/'retrieval.jsonl')
    gold=json.loads((source/'gold-metadata.json').read_text())
    selected=[q for group in ('sage_short','sage_open','litqa2') for q in [q for q in queries if q.id.startswith(group)][:2]]
    config=Config();rows=[];raw=[]
    async with httpx.AsyncClient(transport=Transport('benchmarks/reports/decision-audit'),timeout=35) as client:
        for q in selected:
            row={'query_id':q.id,'retrieval':{}}
            for method in ('papers','snippets'):
                adapter=SemanticScholar(client,config.api_keys.get('semantic_scholar',''),'scholar')
                start=perf_counter()
                try:
                    docs=await (adapter.search(q.question,10) if method=='papers' else adapter.search_snippets(q.question,10))
                    ranked=fuse(docs)
                    if q.id.startswith('litqa2'):
                        ranked=[r for r in ranked if r.year is not None and r.year<2024]
                    row['retrieval'][method]={'status':'complete','unique_papers':len(ranked),
                        'body_snippets':sum(m.get('snippet_kind')=='body' for r in ranked for m in r.passage_metadata),
                        'metrics':score(q,ranked,gold)}
                    raw.append({'query_id':q.id,'method':method,'results':[asdict(r) for r in ranked]})
                    if method=='snippets':row['retrieval'][method].update(retrieval_version=adapter.snippet_version,warnings=adapter.snippet_warnings)
                except Exception as error:
                    row['retrieval'][method]={'status':'failed','error_type':type(error).__name__,
                        'http_status':getattr(getattr(error,'response',None),'status_code',None)}
                row['retrieval'][method]['seconds']=perf_counter()-start
            candidates=deepcopy(q.candidates[:5])
            before=deepcopy(candidates)
            for r in before:prepare(r,q.question)
            async def bounded(op):return await op
            detail=await enrich(client,config,q.question,candidates,bounded)
            for r in candidates:prepare(r,q.question)
            detail['selected_body_passages']=sum(m.get('snippet_kind')=='body' for r in candidates for m in r.passage_metadata)
            detail['changed_papers']=sum(a.passages!=b.passages for a,b in zip(before,candidates))
            detail['output_tokens']=evidence_tokens(budget_results(candidates,5,1500))
            row['enrichment']=detail
            raw.append({'query_id':q.id,'method':'enriched_shortlist','results':[asdict(r) for r in candidates]})
            rows.append(row)
            print(json.dumps({'query':q.id,'retrieval':{k:v['status'] for k,v in row['retrieval'].items()},'enrichment':detail['status'],'enriched':detail['enriched_papers']}),flush=True)
    ROOT.mkdir(exist_ok=True,parents=True)
    (ROOT/'raw.private.json').write_text(json.dumps(raw))
    (ROOT/'per-query.private.json').write_text(json.dumps(rows,indent=2))
    report(rows)


def report(rows):
    summary={'queries':len(rows),'retrieval':{},'enrichment':{},'limitations':[
        'Six reused questions; not independent validation or an established improvement',
        'Live provider results can change; raw normalized snapshots retained privately',
        'Paper search uses existing ten-term lexical cleanup; snippets use the full question with an explicit literal-asterisk HTTP-400 retry. This changes query strategy as well as index',
        'Equal limit=10 counts papers in paper search but snippets in snippet search; fewer unique snippet papers are possible',
        'LitQA historical filtering conservatively excludes 2024 and later papers; no publication-date lookup in this pilot',
        'No model-based relevance or passage usefulness assessment; key lifetime limit blocks fresh model calls',
        'Enrichment uses a fixed five-paper frozen shortlist and one restricted request; papers may receive no snippet',
        'Body-text availability and changed excerpts are coverage measurements, not evidence-quality judgments']}
    for method in ('papers','snippets'):
        success=[r for r in rows if r['retrieval'][method]['status']=='complete']
        summary['retrieval'][method]={'successful':len(success),'unique_papers':sum(r['retrieval'][method]['unique_papers'] for r in success),
            'body_snippets':sum(r['retrieval'][method]['body_snippets'] for r in success)}
    for group,metric in [('sage_short','Hit@10'),('sage_open','WeightedRecall@10'),('litqa2','Recall@30')]:
        paired=[r for r in rows if r['query_id'].startswith(group) and all(r['retrieval'][m]['status']=='complete' for m in ('papers','snippets'))]
        for method in ('papers','snippets'):
            summary['retrieval'][method][group]={'paired_queries':len(paired),'metric':metric,
                'mean':mean(r['retrieval'][method]['metrics'][metric] for r in paired) if paired else None}
    success=[r['enrichment'] for r in rows if r['enrichment']['status']=='complete']
    summary['enrichment']={'successful':len(success),**{key:sum(r[key] for r in success) for key in ('requested_papers','enriched_papers','selected_body_passages','changed_papers')}}
    (ROOT/'summary.json').write_text(json.dumps(summary,indent=2))
    lines=['# Semantic Scholar snippet pilot','',f"Six-query live pilot: {summary['retrieval']['papers']['successful']}/6 paper searches, {summary['retrieval']['snippets']['successful']}/6 snippet searches succeeded.",'',
        '| Method | Unique papers, summed per query | Body snippets |','|---|---:|---:|']
    for method,d in summary['retrieval'].items():lines.append(f"| {method} | {d['unique_papers']} | {d['body_snippets']} |")
    lines+=['', '| Sample / metric | Paired queries | Paper search | Snippet search |', '|---|---:|---:|---:|']
    for group in ('sage_short','sage_open','litqa2'):
        a,b=(summary['retrieval'][m][group] for m in ('papers','snippets'))
        lines.append(f"| {group} / {a['metric']} | {a['paired_queries']} | {a['mean']} | {b['mean']} |")
    lines+=['', 'Enrichment: '+json.dumps(summary['enrichment']), '',*['- '+x for x in summary['limitations']], '',
        'Reproduce: `python -m benchmarks.snippet_eval` with private academic snapshots and a Semantic Scholar key.',
        'API contract: https://api.semanticscholar.org/api-docs/snippets']
    (ROOT/'README.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':asyncio.run(run())
