"""Evaluate useful scholar diversity, separately from provider-count cosmetics."""
from collections import Counter
from copy import deepcopy
import csv
from dataclasses import asdict
import json
from pathlib import Path
from statistics import mean
from slingsearch.diversity import balanced_shortlist, diverse_order
from slingsearch.fusion import fuse
from slingsearch.models import Result
from slingsearch.output import budget_results, evidence_tokens
from slingsearch.passages import select
from .evaluate import bootstrap_delta, document_id, metrics


def positive_ids(results, qrels):
    return {document_id(r, qrels) for r in results if qrels.get(document_id(r, qrels), 0) > 0}


def assess(query, pool, ranked):
    exa = [r for r in pool if any(s.provider == 'exa' for s in r.sources)]
    available = positive_ids(pool, query.qrels) - positive_ids(exa, query.qrels)
    output = budget_results(ranked, 5, 1500)
    found = positive_ids(output, query.qrels)
    new = found & available
    return {**metrics(output, query.qrels),
            'new_relevant_at5': len(new), 'novel_relevant_available': len(available),
            'novel_relevant_retained': len(new), 'output_tokens': evidence_tokens(output),
            'output_count': len(output), 'result_ids': [r.id for r in output],
            'paper_origins': sorted({s.provider for r in output for s in r.sources})}


def evaluate(queries, scored_rows, manifest):
    scored = {r['query_id']: r for r in scored_rows}
    heldout = {'litsearch-' + str(i) for i in manifest['heldout_indices']}
    rows, audits = [], []
    for q in queries:
        pool = fuse(q.candidates)
        for paper in pool:
            paper.passages = select(q.question, [paper.abstract] if paper.abstract else paper.passages)
        exa = sorted([r for r in pool if any(s.provider == 'exa' for s in r.sources)],
                     key=lambda r: min(s.rank for s in r.sources if s.provider == 'exa'))
        split = 'heldout' if q.id in heldout else 'development'
        baseline = {'exa': exa, 'rrf': pool[:20]}
        score = scored.get(q.id)
        if score and score.get('success'):
            ranked = [Result.from_dict(r) for r in score['results']]
            allowed_rrf = {r.id for r in pool[:20]}
            allowed_balanced = {r.id for r in balanced_shortlist(pool, 20)}
            plain = [r for r in ranked if r.id in allowed_balanced]
            baseline.update(global_rrf=[r for r in ranked if r.id in allowed_rrf], global_balanced=plain)
            for band in (.05, .1, .2):
                baseline[f'diverse_{band}'] = diverse_order(plain, band_fraction=band)
            # An Exa-view control uses the identical document representations and
            # global judgments. It isolates adding documents, not enrichment.
            exa_ids = {r.id for r in exa}
            baseline['global_exa_view'] = [r for r in ranked if r.id in exa_ids]
        for method, ranking in baseline.items():
            row = {'query_id': q.id, 'split': split, 'method': method, **assess(q, pool, ranking)}
            rows.append(row)
        exa_positive = positive_ids(exa, q.qrels)
        for r in pool:
            doc_id = document_id(r, q.qrels)
            if q.qrels.get(doc_id, 0) > 0 and doc_id not in exa_positive:
                audits.append({'query_id': q.id, 'split': split, 'question': q.question,
                               'paper': r.title, 'url': r.url, 'providers': sorted({s.provider for s in r.sources}),
                               'canonical_id': r.id, 'relevance_id': doc_id,
                               'retained_by': [row['method'] for row in rows if row['query_id'] == q.id and r.id in row['result_ids']]})
    dev = [r for r in rows if r['split'] == 'development']
    methods = sorted({r['method'] for r in rows})
    # Select only on development labels. Diversity must retain quality and improve
    # known novel-paper retention; otherwise choose relevance-only selection.
    averages = {m: mean(r['nDCG@5'] for r in dev if r['method'] == m) for m in methods if any(r['method'] == m for r in dev)}
    reference = averages.get('global_balanced', 0)
    baseline_new = sum(r['new_relevant_at5'] for r in dev if r['method'] == 'global_balanced')
    eligible = [m for m in ['diverse_0.05','diverse_0.1','diverse_0.2']
                if m in averages and averages[m] >= reference - .01
                and sum(r['new_relevant_at5'] for r in dev if r['method'] == m) > baseline_new]
    chosen = max(eligible, key=lambda m: (sum(r['new_relevant_at5'] for r in dev if r['method'] == m), averages[m])) if eligible else ('global_balanced' if 'global_balanced' in averages else None)
    summary = {}
    for split in ('development','heldout'):
        subset = [r for r in rows if r['split'] == split]
        summary[split] = {}
        for method in methods:
            values = [r for r in subset if r['method'] == method]
            if not values:
                continue
            available = sum(r['novel_relevant_available'] for r in values)
            retained = sum(r['novel_relevant_retained'] for r in values)
            summary[split][method] = {'queries': len(values),
                **{metric: mean(r[metric] for r in values) for metric in ('nDCG@5','R@5','R@10','new_relevant_at5','output_tokens','output_count')},
                'novel_relevant_available': available, 'novel_relevant_retained': retained,
                'novel_relevant_retention': retained / available if available else None}
        baseline = {r['query_id']: r['nDCG@5'] for r in subset if r['method'] == 'exa'}
        for method in methods:
            values = {r['query_id']: r['nDCG@5'] for r in subset if r['method'] == method}
            if values and method != 'exa':
                summary[split][method]['paired_vs_exa'] = bootstrap_delta(values, baseline)
    return {'summary': summary, 'selected_on_development': chosen, 'records': rows, 'novel_paper_audit': audits,
            'successful_global_rankings': sum(r.get('success', False) for r in scored_rows),
            'failed_global_rankings': [r['query_id'] for r in scored_rows if not r.get('success')],
            'limitations': 'Known-paper qrels are incomplete; only annotated non-Exa papers count as useful diversity. Shared identities avoid false recall gains. Actual output obeys 1500 estimated tokens, with at most five results.'}


def save_report(result, out):
    out = Path(out)
    (out/'comparison.json').write_text(json.dumps(result, indent=2))
    lines = ['# Scholar diversity: retrieval contribution and final retention','',
             'Known-paper relevance, actual output capped at five results / 1,500 estimated tokens. No final provider quotas. Shared identities and global judgments are used across comparisons.','',
             f'Diversity policy selected on development queries: `{result["selected_on_development"]}`. Held-out labels did not select it.','']
    for split, methods in result['summary'].items():
        lines += [split, '', '| Method | Queries | nDCG@5 | Recall@5 | New relevant papers retained / available | Mean output tokens |', '|---|---:|---:|---:|---:|---:|']
        for method, values in methods.items():
            lines.append(f'| {method} | {values["queries"]} | {values["nDCG@5"]:.4f} | {values["R@5"]:.4f} | {values["novel_relevant_retained"]} / {values["novel_relevant_available"]} | {values["output_tokens"]:.0f} |')
        lines.append('')
    lines += ['Novel papers not found by Exa:', '']
    for r in result['novel_paper_audit']:
        lines.append(f'- {r["query_id"]} ({r["split"]}): [{r["paper"]}]({r["url"]}), found by {", ".join(r["providers"])}.')
    lines += ['', result['limitations'], '', 'Model scores were computed once over the complete unique pool (up to 40 papers), then reused across candidate-selection policies. Independent Decisions judgments make this reuse exact at the request level. Main pool comparisons use the same top-20 count and 300-token document representation. Provider failures remain in the collection snapshot.']
    (out/'report.md').write_text('\n'.join(lines)+'\n')


def main():
    import argparse
    from .datasets import load_snapshot
    parser = argparse.ArgumentParser(description='Re-evaluate saved scholar judgments without API calls')
    parser.add_argument('--snapshot', required=True)
    parser.add_argument('--scores', required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    queries = load_snapshot(args.snapshot)
    scored = [json.loads(line) for line in Path(args.scores).read_text().splitlines() if line.strip()]
    manifest = json.loads(Path(args.manifest).read_text())
    Path(args.output).mkdir(parents=True, exist_ok=True)
    save_report(evaluate(queries, scored, manifest), args.output)
    print('Rebuilt comparisons from saved data; no API requests.')


if __name__ == '__main__':
    main()
