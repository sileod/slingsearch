"""Paired reports for the frozen add-Expand experiment."""
from copy import deepcopy
from dataclasses import asdict
import json
from statistics import mean,median
from pathlib import Path
from benchmarks.expand_ablation import ROOT,METHODS,candidates_for,evidence_key,digest
from benchmarks.datasets import load_snapshot
from benchmarks.academic_eval import score,lookup_id,aliases,gold_aliases
from benchmarks.evaluate import bootstrap_delta
from slingsearch import Result
from slingsearch.fusion import normalize,normalize_doi


def resolve_for_scoring(documents,gold):
    docs=deepcopy(documents)
    for r in docs:
        paper=gold.get(lookup_id(r))
        if paper:
            r.identifiers.update(aliases(paper));r.doi=r.doi or r.identifiers.get('doi');normalize(r)
    return docs


def reference_hits(q,documents,gold):
    mapping=gold_aliases(q,gold);found=set()
    for r in resolve_for_scoring(documents,gold):
        keys=[r.id]+[k+':'+(normalize_doi(v) if k=='doi' else v) for k,v in r.identifiers.items()]
        if r.doi:keys.append('doi:'+normalize_doi(r.doi))
        found.update(mapping[k] for k in keys if k in mapping)
    return found


def report():
    manifest=json.loads((ROOT/'manifest.json').read_text())
    assert digest(ROOT/'frozen.jsonl')==manifest['frozen_sha256']
    queries={q.id:q for q in load_snapshot(ROOT/'frozen.jsonl')};gold=json.loads((ROOT/'gold-metadata.private.json').read_text())
    grades={}
    if (ROOT/'web-judgments.private.json').exists():
        mapping=json.loads((ROOT/'blind-map.private.json').read_text());judgments=json.loads((ROOT/'web-judgments.private.json').read_text())
        for name,row in judgments.items():grades[mapping[name]['evidence_key']]=row['grade']
    runs={}
    for backend,file in [('rrf','rrf-rankings.private.jsonl'),('span_free','free-rankings.private.jsonl')]:
        p=ROOT/file
        if not p.exists():continue
        for line in p.read_text().splitlines():
            row=json.loads(line)
            key=(backend,row['query_id'],row['method'])
            if row['success'] or key not in runs:runs[key]=row
    per_query=[];missing_grades=0
    for (backend,qid,method),row in runs.items():
        q=queries[qid]
        record={'query_id':qid,'mode':q.mode,'method':method,'backend':backend,'success':row['success']}
        if row['success']:
            output=[Result.from_dict(r) for r in row['output']]
            record.update(returned=len(output),output_tokens=row['output_tokens'])
            if q.mode=='web':
                keys=[evidence_key(qid,r) for r in row['output']]
                if all(key in grades for key in keys):
                    values=[grades[key] for key in keys]
                    record.update(grade=mean(values) if values else 0,
                        useful_precision=sum(v>=2 for v in values)/len(values) if values else 0,
                        direct_query=float(any(v==3 for v in values)),
                        useful_per_1000_tokens=1000*sum(v>=2 for v in values)/row['output_tokens'] if row['output_tokens'] else 0)
                else:missing_grades+=sum(key not in grades for key in keys)
            else:
                record['metrics']=score(q,resolve_for_scoring(output,gold),gold)
                _,shortlist=candidates_for(q,method);lookup={r.id:r for r in shortlist}
                order=[lookup[i] for i in row['shortlist_ids'] if i in lookup]
                record['ranking_metrics']=score(q,resolve_for_scoring(order,gold),gold)
        per_query.append(record)
    if missing_grades:
        print(json.dumps({'status':'awaiting_web_judgments','missing_output_occurrences':missing_grades}))
        return
    dump={'manifest':manifest,'web':{},'scholar':{},'paired':{},'retrieval':{},'missing_web_grades':missing_grades,
        'limitations':[
            'Eight web questions and eight scholar questions: exploratory, low statistical power',
            'Held out from 208 prior local experiment questions, not guaranteed held out from provider training',
            'FreshQA stable-fact adaptation evaluates evidence excerpts, not FreshQA answer generation',
            'Web labels come from an independent free-model blinded audit, not humans or full-page verification',
            'The eight-item assistant spot-check agreed on useful/not-useful labels but only half of exact ordinal grades; at least one rationale inferred a game name beyond the displayed excerpt. Fine-grained grade effects should be treated cautiously',
            'Span-01 Lite binary evidence scores are not numerically equivalent to Jev ordinal rubric scores',
            'Jev is blocked by the existing key cap. This does not validate the shipped Jev default',
            'SAGE reference labels are incomplete; known-paper coverage does not measure all academic relevance',
            'All arms share frozen base retrieval and identical limits; Expand adds one retrieval call',
            'Source identity enrichment is used for scoring only; no target abstracts or answers enter ranking',
            'Latency is retrieval-only (maximum constituent provider latency); shared free-model scoring is amortized and not a production latency comparison']}
    for backend in ('rrf','span_free'):
        for method in METHODS:
            rows=[r for r in per_query if r['backend']==backend and r['method']==method and r['mode']=='web' and r['success'] and 'grade' in r]
            if rows:dump['web'][backend+':'+method]={'queries':len(rows),**{k:mean(r[k] for r in rows) for k in ['grade','useful_precision','direct_query','useful_per_1000_tokens','returned','output_tokens']}}
            for kind in ('short','open'):
                rows=[r for r in per_query if r['backend']==backend and r['method']==method and r['query_id'].startswith('sage_'+kind) and r['success']]
                if rows:dump['scholar'][backend+':'+method+':'+kind]={'queries':len(rows),
                    **{k:mean(r['metrics'][k] for r in rows) for k in ['nDCG@5','Hit@5','WeightedRecall@5']},
                    'ranking_recall10':mean(r['ranking_metrics']['Recall@10'] for r in rows),
                    'returned':mean(r['returned'] for r in rows)}
        for metric in ('grade','useful_precision','direct_query','useful_per_1000_tokens'):
            expanded={r['query_id']:r[metric] for r in per_query if r['backend']==backend and r['method']=='default+expand' and r['mode']=='web' and metric in r}
            baseline={r['query_id']:r[metric] for r in per_query if r['backend']==backend and r['method']=='default' and r['mode']=='web' and metric in r}
            if expanded and baseline:dump['paired'][backend+':web:'+metric]=bootstrap_delta(expanded,baseline)
    for mode in ('web','scholar'):
        qs=[q for q in queries.values() if q.mode==mode]
        d={'queries':len(qs),'provider_success':{},'latency_p50':{},'expand_unique_known_positive_pairs':0}
        names=manifest['defaults'][mode]+['expand']
        d['provider_success']={n:sum(q.retrieval[n]['success'] for q in qs) for n in names}
        for method in METHODS:
            selected=['expand'] if method=='expand' else manifest['defaults'][mode]+(['expand'] if method.endswith('+expand') else [])
            d['latency_p50'][method]=median(max(q.retrieval[n]['seconds'] for n in selected) for q in qs)
        if mode=='scholar':
            for kind in ('short','open'):
                for method in METHODS:
                    values=[];shortlisted=[]
                    for q in qs:
                        if not q.id.startswith('sage_'+kind):continue
                        pool,shortlist=candidates_for(q,method)
                        values.append(score(q,resolve_for_scoring(pool,gold),gold))
                        shortlisted.append(score(q,resolve_for_scoring(shortlist,gold),gold))
                    metric='Hit@10' if kind=='short' else 'WeightedRecall@10'
                    d[kind+':'+method]={'metric':metric,'pool_known_coverage':mean(v['Recall@30'] for v in values),
                        'shortlist_score':mean(v[metric] for v in shortlisted)}
            for q in qs:
                base,_=candidates_for(q,'default');expanded,_=candidates_for(q,'default+expand')
                d['expand_unique_known_positive_pairs']+=len(reference_hits(q,expanded,gold)-reference_hits(q,base,gold))
            d['unresolved_reference_ids']=sum(pid.removeprefix('semantic_scholar:') not in gold for q in qs for pid in q.qrels)
        dump['retrieval'][mode]=d
    spot_path=ROOT/'assistant-spotcheck-grades.private.json'
    if spot_path.exists() and (ROOT/'web-judgments.private.json').exists():
        spot=json.loads(spot_path.read_text());judgments=json.loads((ROOT/'web-judgments.private.json').read_text())
        pairs=[(v['grade'],judgments[k]['grade']) for k,v in spot.items() if k in judgments]
        dump['judge_spotcheck']={'questions':len(pairs),'exact_grade_agreement':mean(a==b for a,b in pairs),
            'useful_binary_agreement':mean((a>=2)==(b>=2) for a,b in pairs),
            'scope':'Assistant audit of one randomly selected blinded excerpt per question, without viewing judge labels first; not human validation'}
    dump['decision']='Keep Expand opt-in: this experiment cannot establish a reliable gain for the shipped Jev default.'
    (ROOT/'summary.json').write_text(json.dumps(dump,indent=2));(ROOT/'per-query-metrics.private.json').write_text(json.dumps(per_query,indent=2))
    lines=['# Expand add-one-provider ablation','',dump['decision'],'',
        '16 newly sampled questions: 8 FreshQA stable true-premise web questions and 8 SAGE questions (one per task/domain stratum). Seed 20261031; 208 previously used local questions excluded. All providers requested 10 results; shortlist 20; candidate text 400 tokens; final maximum 5 sources and 1,500 tokens.',
        '', '## Web: actual displayed evidence','',
        'Independent free Nemotron judgment, blinded to provider/method/rank. Grade 0 unrelated or contradictory, 1 topical only, 2 partial supporting evidence, 3 direct answer evidence. Mean grade is among returned sources; empty output scores zero. These are model estimates, not human judgments.', '',
        '| Ranking / retrieval | Queries | Mean grade (0–3) | Useful precision | Direct-evidence query coverage | Useful sources / 1k tokens | Returned |', '|---|---:|---:|---:|---:|---:|---:|']
    for name,d in dump['web'].items():lines.append(f"| {name} | {d['queries']} | {d['grade']:.3f} | {d['useful_precision']:.3f} | {d['direct_query']:.3f} | {d['useful_per_1000_tokens']:.3f} | {d['returned']:.2f} |")
    lines+=['','Paired differences are **default+Expand minus default**:','']
    for name,d in dump['paired'].items():lines.append(f"- {name}: {d['mean_delta']:+.3f}, query-bootstrap 95% CI [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}], n={d['queries']}.")
    lines+=['','## Scholar: known reference papers','', '| Ranking / retrieval / task | Queries | nDCG@5 | Hit@5 | WeightedRecall@5 | Recall@10 after ranking/cutoff | Returned |','|---|---:|---:|---:|---:|---:|---:|']
    for name,d in dump['scholar'].items():lines.append(f"| {name} | {d['queries']} | {d['nDCG@5']:.3f} | {d['Hit@5']:.3f} | {d['WeightedRecall@5']:.3f} | {d['ranking_recall10']:.3f} | {d['returned']:.2f} |")
    lines+=['','## Retrieval and operational scope','']
    for mode,d in dump['retrieval'].items():
        lines.append(f"- {mode}: provider success {d['provider_success']}; retrieval p50 seconds {d['latency_p50']}.")
        if mode=='scholar':lines.append(f"- Expand contributed {d['expand_unique_known_positive_pairs']} additional known-positive query-paper pairs; unresolved reference IDs: {d['unresolved_reference_ids']}.")
    if 'judge_spotcheck' in dump:lines.append('- Judge spot-check: '+json.dumps(dump['judge_spotcheck']))
    lines+=['- Incremental Expand search cost: documented $1.50 per 1,000 calls; provider-reported USD cost unavailable. Free Span and independent judge report zero model cost.',
        '', '## Limitations','',*['- '+v for v in dump['limitations']], '',
        'Reproduce with retained private snapshots: `python -m benchmarks.expand_ablation rrf`, `python -m benchmarks.expand_ablation free`, `python -m benchmarks.expand_ablation blind`, `python -m benchmarks.expand_judge`, `python -m benchmarks.expand_report`. Fresh collection requires the pinned FreshQA CSV and SAGE release plus configured credentials. Paid calls share the persistent session budget.',
        '', 'Sources: [FreshQA](https://github.com/freshllms/freshqa), [SAGE](https://github.com/HughieHu/Sage), [free Span](https://openrouter.ai/respan/span-01-lite:free), [Expand](https://expand.ai/docs/search).']
    (ROOT/'README.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'web':dump['web'],'paired':dump['paired'],'missing_grades':missing_grades},indent=2))


if __name__=='__main__':report()
