"""Select on exposed development queries; evaluate once on unused frozen queries."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from statistics import mean
from .sample import web_sample
from .anchor import anchor
from .evaluate import bootstrap_delta, document_id, metrics
from slingsearch.fusion import fuse


def evaluate(queries, method):
    return {q.id: metrics(method(q.candidates), q.qrels) for q in queries}


def experiment(queries, development_ids, excluded_ids=()):
    queries = [q for q in queries if q.id not in set(excluded_ids)]
    dev = [q for q in queries if q.id in development_ids]
    holdout = [q for q in queries if q.id not in development_ids]
    if not dev or not holdout:
        raise ValueError('Both development and untouched holdout queries are required')
    # Fixed small grid; holdout labels do not select providers or parameters.
    grid = [(keep, weight) for keep in (1, 2) for weight in (0, .25, .5, 1, 2)]
    trials = []
    for keep, weight in grid:
        scores = evaluate(dev, lambda rows: anchor(rows, keep=keep, weight=weight))
        trials.append({'keep': keep, 'weight': weight,
                       'development_nDCG@5': mean(s['nDCG@5'] for s in scores.values())})
    chosen = max(trials, key=lambda r: r['development_nDCG@5'])
    def individual(rows):
        pool = fuse([r for r in rows if any(s.provider == 'exa' for s in r.sources)])
        return sorted(pool, key=lambda r: min(s.rank for s in r.sources if s.provider == 'exa'))
    def trio_rrf(rows):
        return fuse([r for r in rows if any(s.provider in ('exa','parallel','brave') for s in r.sources)])
    methods = {'exa': individual, 'rrf_trio': trio_rrf,
               'anchor_selected_on_dev': lambda rows: anchor(rows, keep=chosen['keep'], weight=chosen['weight'])}
    results = {}
    for name, sample in [('development', dev), ('holdout', holdout)]:
        runs = {method: evaluate(sample, rank) for method, rank in methods.items()}
        results[name] = {'queries': len(sample), 'query_ids': [q.id for q in sample],
                         'metrics': {m: {metric: mean(s[metric] for s in scores.values())
                                         for metric in ('nDCG@5','nDCG@10','R@10')} for m,scores in runs.items()},
                         'paired_vs_exa': {m: bootstrap_delta({q:s['nDCG@5'] for q,s in scores.items()},
                                                             {q:s['nDCG@5'] for q,s in runs['exa'].items()})
                                             for m,scores in runs.items() if m != 'exa'}}
    return {'selection': chosen, 'development_trials': trials, 'results': results,
            'limitations': 'Frozen Web Search Benchmark Kit heuristic labels; holdout is query-disjoint within one benchmark, not independent human validation.'}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--results', required=True);p.add_argument('--judgments', required=True)
    p.add_argument('--queries', required=True);p.add_argument('--runs', required=True)
    p.add_argument('--development-manifest', required=True);p.add_argument('--output', required=True)
    p.add_argument('--exclude-ids', nargs='*', default=[], help='Other previously exposed queries excluded from holdout')
    args=p.parse_args()
    queries,info=web_sample(args.results,args.judgments,args.queries,args.runs,size=10000)
    development=json.loads(Path(args.development_manifest).read_text())['selected_ids']
    result=experiment(queries,set(development),args.exclude_ids);result['sampling']=info
    result['excluded_previously_exposed_ids']=args.exclude_ids
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    (out/'results.json').write_text(json.dumps(result,indent=2))
    (out/'frozen.jsonl').write_text(''.join(json.dumps(asdict(q))+'\n' for q in queries))
    lines=['# Exa-preserving fusion: development and query-held-out evaluation','',
           'No paid requests. Parameters selected on the previously exposed 20 queries; earlier smoke-check questions KI02/KI03 are also excluded from holdout. Heuristic relevance labels limit the conclusion.','',
           f'Selected rule: preserve Exa top {result["selection"]["keep"]}; supplementary rank-support weight {result["selection"]["weight"]}.','',
           '| Split | Queries | Exa nDCG@5 | RRF trio | Selected anchor |','|---|---:|---:|---:|---:|']
    for name,r in result['results'].items():
        m=r['metrics'];lines.append(f'| {name} | {r["queries"]} | {m["exa"]["nDCG@5"]:.4f} | {m["rrf_trio"]["nDCG@5"]:.4f} | {m["anchor_selected_on_dev"]["nDCG@5"]:.4f} |')
    lines+=['','Paired holdout intervals vs Exa:','',json.dumps(result['results']['holdout']['paired_vs_exa'],indent=2)]
    (out/'report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
