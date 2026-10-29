# Sampled DORIS-MAE ranking evaluation

16 paired queries; full released candidate pools; identical 400-estimated-token document representations.

| Method | nDCG@5 | nDCG@10 | Recall@10 | Aspect coverage@5 | Latency p50 (s) | Ranking USD/1000 |
|---|---:|---:|---:|---:|---:|---:|
| bm25 | 0.5806 | 0.5946 | 0.1571 | 0.7506 | 0.03 | 0.000 |
| random | 0.4986 | 0.5084 | 0.1018 | 0.7185 | 0.01 | 0.000 |
| jev | 0.7324 | 0.7503 | 0.3068 | 0.7668 | 25.28 | 3.825 |
| decider | 0.7537 | 0.7608 | 0.2889 | 0.7752 | 25.55 | 1.290 |
| voyage | 0.7668 | 0.7817 | 0.3523 | 0.8023 | 0.85 | 0.860 |

## Interpretation and limits

- Ranking-only test: these are native dataset candidates, not live Exa or blended provider results. No provider-fusion or diversity win can be inferred.
- Labels are released AnnoGPT judgments, not exhaustive independent human gold. Linear summed gains match upstream DORIS normal nDCG. Recall uses upstream mean-aspect-score >= 1.
- Recall excludes queries with no papers above that threshold, following the upstream convention. Per-metric query counts are recorded in results.json.
- Aspect coverage is a diagnostic: sum of best selected judgment per aspect divided by the same quantity over the whole pool. It can reward partial papers and must be read alongside relevance.
- BM25 sees the same bounded document text as the models. Random is one fixed seeded permutation per query. Neither baseline uses labels.
- Long abstracts are truncated to a shared 400 estimated tokens, preserving literal prefixes. The labels were assigned to full abstracts, so relevant evidence beyond the cutoff can penalize every method. This is a budget-constrained test, not an unbounded full-abstract result.
- Table measures document ordering; results.json separately includes the actual five-result/1500-token output metrics.
- Ranking cost/1000 extrapolates full native pools of roughly 100 papers, not the product default 20-candidate shortlist. It excludes retrieval and any aborted or failed requests; spend.json tracks all attempted requests separately.
- Query-level paired bootstrap intervals and full rankings are saved in results.json and rankings.jsonl. Small-sample results are provisional.
- Runtime concurrency was increased during the run (12 to 32); one question resumes cached judgments. Latency figures describe observed execution, not a controlled speed benchmark. See manifest runtime_note.
- PaperFindingBench is a separate retrieval evaluation; do not describe this ranking-only run as a PaperFindingBench or provider-fusion result.

## Reproduce

Free replay: `python -m benchmarks.scholar_eval report --out benchmarks/reports/scholar-benchmarks/doris`

Paid run/resume: set OPENROUTER_API_KEY, then `python -m benchmarks.scholar_eval run --out benchmarks/reports/scholar-benchmarks/doris`. Completed query/method pairs are reused. Reservations persist across restarts.
