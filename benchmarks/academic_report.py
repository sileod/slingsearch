"""Free paired reports for the original and expanded academic source snapshots."""
import json
from pathlib import Path
from statistics import mean
from benchmarks.datasets import load_snapshot
from benchmarks.evaluate import bootstrap_delta
from benchmarks.academic_eval import score


def report(root):
    root = Path(root)
    queries = load_snapshot(root / 'retrieval.jsonl')
    gold = json.loads((root / 'gold-metadata.json').read_text())
    records = {}
    for line in (root / 'rankings.jsonl').read_text().splitlines():
        row = json.loads(line)
        records[row['query_id'], row['method']] = row
    providers = sorted({p for q in queries for p in q.retrieval if p in ('exa','parallel','openalex','semantic_scholar','serpapi','papers_with_code','huggingface')})
    output = {'groups': {}, 'source_scope': root.name,
        'protocol': {'per_provider': 10, 'shortlist_max': 50, 'candidate_tokens': 400, 'product_results': 5, 'product_tokens': 1500},
        'limitations': ['Live API adaptation, not the official closed-corpus leaderboard',
            'Known-positive ID/DOI matching only; SAGE reference positives are incomplete',
            'Single providers request only ten results; Recall@30 is depth-limited for those baselines',
            'Main table uses paired complete ranking cases; provider retrieval failure is counted as zero in operational precision/recall',
            'Search CLI comparison is a Python fusion port with fixed alphabetic bucket ties, not execution of the Rust CLI',
            'Expanded snapshot adds sources and deterministic lexical query cleanup; not a fresh-query holdout',
            'No answer correctness or passage usefulness claim follows from finding a paper']}
    markdown = ['# Sampled academic retrieval: ' + root.name, '',
        'SAGE: 12 known-paper and 12 open-ended questions across four disciplines. LitQA: all 10 official validation questions. Source revisions and selected IDs were fixed before retrieval. This is an open-web adaptation, not an official leaderboard reproduction.', '',
        'All document metrics use exact paper IDs/DOIs enriched before ranking. Gold answers, source titles and SAGE generation plans are never appended to queries. The budgeted product view is scored separately from the full ordered list.', '']
    for group, prefix, primary in [('SAGE known paper', 'sage_short', 'Hit@10'), ('SAGE open-ended', 'sage_open', 'WeightedRecall@10'), ('LitQA full-text search', 'litqa2', 'Recall@30')]:
        qs = [q for q in queries if q.id.startswith(prefix)]
        if not qs: continue
        methods = sorted({m for qid, m in records if qid in {q.id for q in qs}})
        required = [m for m in ['rrf','voyage','jev_packed5','d1_packed5','jev_voyage','d1_voyage','jev_d1_voyage'] if m in methods]
        paired_ids = {q.id for q in qs if all(records.get((q.id, m), {}).get('success') for m in required)}
        summaries, paired = {}, {}
        for method in methods:
            success_rows = [records[q.id, method] for q in qs if records.get((q.id, method), {}).get('success')]
            rows = [records[qid, method] for qid in sorted(paired_ids) if records.get((qid, method), {}).get('success')]
            # Retrieval failures count as zero for the paired single-provider baseline.
            retrieval_method = method.startswith('single:') or '+voyage' in method
            denominator = len(paired_ids) if retrieval_method else len(rows)
            summaries[method] = {'successful_queries': len(success_rows), 'total_queries': len(qs), 'paired_queries': len(paired_ids),
                'primary_paired': sum(r['metrics'][primary] for r in rows) / denominator if denominator else None,
                'full_sample_score': sum(r['metrics'][primary] for r in success_rows) / len(qs) if method.startswith('single:') or method in ('rrf','search_cli_fusion_port') else None,
                'product_hit5_paired': sum(r['product_metrics']['Hit@5'] for r in rows) / denominator if denominator else None,
                'all_successful_mean': mean(r['metrics'][primary] for r in success_rows) if success_rows else None}
        baseline_names = [p + '+voyage' for p in providers if p + '+voyage' in methods]
        if baseline_names and paired_ids:
            best = max(baseline_names, key=lambda m: summaries[m]['primary_paired'] or 0)
            for method in required:
                left = {qid: records[qid, method]['metrics'][primary] for qid in paired_ids}
                right = {qid: records[qid, best]['metrics'][primary] if records.get((qid,best),{}).get('success') else 0 for qid in paired_ids}
                paired[method] = bootstrap_delta(left, right)
        else: best = None
        pool = {q.id: score(q, q.candidates, gold) for q in qs}
        per_domain = {}
        for domain in sorted({q.retrieval['domain'] for q in qs if 'domain' in q.retrieval}):
            subset = [q for q in qs if q.retrieval.get('domain') == domain]
            per_domain[domain] = {'queries': len(subset), 'methods': {m: {'successes': sum(records.get((q.id,m),{}).get('success',False) for q in subset),
                'mean': mean(records[q.id,m]['metrics'][primary] if records.get((q.id,m),{}).get('success') else 0 for q in subset)} for m in methods}}
        output['groups'][prefix] = {'queries': len(qs), 'primary_metric': primary, 'paired_ids': sorted(paired_ids),
            'methods': summaries, 'strongest_reranked_individual': best, 'paired_vs_best_single': paired,
            'provider_successes': {p: sum(q.retrieval[p]['success'] for q in qs) for p in providers},
            'pool_recall_at30': mean(v['Recall@30'] for v in pool.values()), 'by_domain': per_domain}
        markdown += ['## ' + group, '', 'Retrieval-only comparison on the **full sample**, independent of model availability:', '',
                     '| Method | Full-sample ' + primary + ' |', '|---|---:|']
        for method, summary in summaries.items():
            if summary['full_sample_score'] is not None:
                markdown.append(f"| {method} | {summary['full_sample_score']:.3f} |")
        markdown += ['', 'The separate ranking table below uses only paired complete model cases. In the expanded source run, upstream Voyage TPM limits exclude many SAGE cases; do not generalize those tiny subsets.', '']
        markdown += [f'Primary metric: **{primary}**. Paired ranking cases: **{len(paired_ids)}/{len(qs)}**.', '',
                     '| Method | Paired score | Successful runs |', '|---|---:|---:|']
        for method, summary in summaries.items():
            value = 'NA' if summary['primary_paired'] is None else f"{summary['primary_paired']:.3f}"
            markdown.append(f"| {method} | {value} | {summary['successful_queries']}/{summary['total_queries']} |")
        if best:
            markdown += ['', f'Strongest reranked individual on this sample: `{best}` (selected by aggregate score, not a per-query oracle). This selection is descriptive and optimistic on a small reused sample.']
            if 'voyage' in paired:
                delta = paired['voyage']
                markdown.append(f"Voyage blend minus that baseline: {delta['mean_delta']:+.3f}, paired bootstrap 95% CI [{delta['ci95'][0]:+.3f}, {delta['ci95'][1]:+.3f}].")
        markdown += ['', 'Provider successes: ' + ', '.join(f"{p} {output['groups'][prefix]['provider_successes'][p]}/{len(qs)}" for p in providers) + '.', '']
    implementations = [q.retrieval.get('implementation_audit', {}) for q in queries]
    output['implementation_coverage'] = {k: sum(r.get(k, 0) for r in implementations) for k in ['github_repositories','papers_with_code_links','papers_with_dataset_links']}
    markdown += ['## Limits and reproduction', '', *['- ' + limitation for limitation in output['limitations']], '',
        'LitQA enforces a conservative publication-date cutoff approximation; this does not verify original corpus insertion membership. Missing date metadata excludes sources. Do not compare these scores directly with the Asta leaderboard.', '',
        'The Search CLI fusion port uses URL-only deduplication and RRF k=60 from pinned upstream source. Provider buckets use fixed alphabetical order for deterministic score ties. This controls retrieval and preserves common metadata/identity matching; it does not reproduce the Rust CLI’s routing, extraction, timing or every runtime tie.', '',
        'GitHub implementation repositories are collected separately and never counted as retrieved papers. Code/task/dataset links are source metadata, not relevance boosts. Coverage totals are query-result counts, not globally unique artifacts.', '',
        'Recompute: `python -m benchmarks.academic_report ' + str(root) + '`. Private snapshots are required; no paid API calls are made by reporting.']
    (root / 'results.json').write_text(json.dumps(output, indent=2))
    (root / 'README.md').write_text('\n'.join(markdown) + '\n')
    print(json.dumps({g: {m: v['primary_paired'] for m,v in r['methods'].items()} for g,r in output['groups'].items()}, indent=2))


if __name__ == '__main__':
    import sys
    report(sys.argv[1])
