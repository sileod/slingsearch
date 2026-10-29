# Sampled academic retrieval: academic-samples

SAGE: 12 known-paper and 12 open-ended questions across four disciplines. LitQA: all 10 official validation questions. Source revisions and selected IDs were fixed before retrieval. This is an open-web adaptation, not an official leaderboard reproduction.

All document metrics use exact paper IDs/DOIs enriched before ranking. Gold answers, source titles and SAGE generation plans are never appended to queries. The budgeted product view is scored separately from the full ordered list.

## SAGE known paper

Retrieval-only comparison on the **full sample**, independent of model availability:

| Method | Full-sample Hit@10 |
|---|---:|
| rrf | 0.167 |
| search_cli_fusion_port | 0.167 |
| single:exa | 0.167 |
| single:openalex | 0.000 |
| single:parallel | 0.083 |
| single:semantic_scholar | 0.000 |

The separate ranking table below uses only paired complete model cases. In the expanded source run, upstream Voyage TPM limits exclude many SAGE cases; do not generalize those tiny subsets.

Primary metric: **Hit@10**. Paired ranking cases: **12/12**.

| Method | Paired score | Successful runs |
|---|---:|---:|
| d1_packed5 | 0.167 | 12/12 |
| d1_voyage | 0.167 | 12/12 |
| exa+voyage | 0.167 | 12/12 |
| jev_d1_voyage | 0.167 | 12/12 |
| jev_packed5 | 0.167 | 12/12 |
| jev_voyage | 0.167 | 12/12 |
| openalex+voyage | 0.000 | 12/12 |
| parallel+voyage | 0.083 | 12/12 |
| rrf | 0.167 | 12/12 |
| search_cli_fusion_port | 0.167 | 12/12 |
| semantic_scholar+voyage | 0.000 | 12/12 |
| single:exa | 0.167 | 12/12 |
| single:openalex | 0.000 | 12/12 |
| single:parallel | 0.083 | 12/12 |
| single:semantic_scholar | 0.000 | 10/12 |
| voyage | 0.167 | 12/12 |

Strongest reranked individual on this sample: `exa+voyage` (selected by aggregate score, not a per-query oracle). This selection is descriptive and optimistic on a small reused sample.
Voyage blend minus that baseline: +0.000, paired bootstrap 95% CI [-0.250, +0.250].

Provider successes: exa 12/12, openalex 12/12, parallel 12/12, semantic_scholar 10/12.

## SAGE open-ended

Retrieval-only comparison on the **full sample**, independent of model availability:

| Method | Full-sample WeightedRecall@10 |
|---|---:|
| rrf | 0.105 |
| search_cli_fusion_port | 0.105 |
| single:exa | 0.069 |
| single:openalex | 0.145 |
| single:parallel | 0.000 |
| single:semantic_scholar | 0.000 |

The separate ranking table below uses only paired complete model cases. In the expanded source run, upstream Voyage TPM limits exclude many SAGE cases; do not generalize those tiny subsets.

Primary metric: **WeightedRecall@10**. Paired ranking cases: **12/12**.

| Method | Paired score | Successful runs |
|---|---:|---:|
| d1_packed5 | 0.062 | 12/12 |
| d1_voyage | 0.096 | 12/12 |
| exa+voyage | 0.069 | 12/12 |
| jev_d1_voyage | 0.086 | 12/12 |
| jev_packed5 | 0.086 | 12/12 |
| jev_voyage | 0.084 | 12/12 |
| openalex+voyage | 0.145 | 12/12 |
| parallel+voyage | 0.000 | 12/12 |
| rrf | 0.105 | 12/12 |
| search_cli_fusion_port | 0.105 | 12/12 |
| semantic_scholar+voyage | 0.000 | 12/12 |
| single:exa | 0.069 | 12/12 |
| single:openalex | 0.145 | 12/12 |
| single:parallel | 0.000 | 12/12 |
| single:semantic_scholar | 0.000 | 8/12 |
| voyage | 0.096 | 12/12 |

Strongest reranked individual on this sample: `openalex+voyage` (selected by aggregate score, not a per-query oracle). This selection is descriptive and optimistic on a small reused sample.
Voyage blend minus that baseline: -0.049, paired bootstrap 95% CI [-0.145, +0.038].

Provider successes: exa 12/12, openalex 12/12, parallel 12/12, semantic_scholar 8/12.

## LitQA full-text search

Retrieval-only comparison on the **full sample**, independent of model availability:

| Method | Full-sample Recall@30 |
|---|---:|
| rrf | 0.600 |
| search_cli_fusion_port | 0.600 |
| single:exa | 0.500 |
| single:openalex | 0.200 |
| single:parallel | 0.100 |
| single:semantic_scholar | 0.000 |

The separate ranking table below uses only paired complete model cases. In the expanded source run, upstream Voyage TPM limits exclude many SAGE cases; do not generalize those tiny subsets.

Primary metric: **Recall@30**. Paired ranking cases: **10/10**.

| Method | Paired score | Successful runs |
|---|---:|---:|
| d1_packed5 | 0.600 | 10/10 |
| d1_voyage | 0.600 | 10/10 |
| exa+voyage | 0.500 | 10/10 |
| jev_d1_voyage | 0.600 | 10/10 |
| jev_packed5 | 0.600 | 10/10 |
| jev_voyage | 0.600 | 10/10 |
| openalex+voyage | 0.200 | 10/10 |
| parallel+voyage | 0.100 | 10/10 |
| rrf | 0.600 | 10/10 |
| search_cli_fusion_port | 0.600 | 10/10 |
| semantic_scholar+voyage | 0.000 | 10/10 |
| single:exa | 0.500 | 10/10 |
| single:openalex | 0.200 | 10/10 |
| single:parallel | 0.100 | 10/10 |
| single:semantic_scholar | 0.000 | 6/10 |
| voyage | 0.600 | 10/10 |

Strongest reranked individual on this sample: `exa+voyage` (selected by aggregate score, not a per-query oracle). This selection is descriptive and optimistic on a small reused sample.
Voyage blend minus that baseline: +0.100, paired bootstrap 95% CI [+0.000, +0.300].

Provider successes: exa 10/10, openalex 10/10, parallel 10/10, semantic_scholar 6/10.

## Limits and reproduction

- Live API adaptation, not the official closed-corpus leaderboard
- Known-positive ID/DOI matching only; SAGE reference positives are incomplete
- Single providers request only ten results; Recall@30 is depth-limited for those baselines
- Main table uses paired complete ranking cases; provider retrieval failure is counted as zero in operational precision/recall
- Search CLI comparison is a Python fusion port with fixed alphabetic bucket ties, not execution of the Rust CLI
- Expanded snapshot adds sources and deterministic lexical query cleanup; not a fresh-query holdout
- No answer correctness or passage usefulness claim follows from finding a paper

LitQA enforces a conservative publication-date cutoff approximation; this does not verify original corpus insertion membership. Missing date metadata excludes sources. Do not compare these scores directly with the Asta leaderboard.

The Search CLI fusion port uses URL-only deduplication and RRF k=60 from pinned upstream source. Provider buckets use fixed alphabetical order for deterministic score ties. This controls retrieval and preserves common metadata/identity matching; it does not reproduce the Rust CLI’s routing, extraction, timing or every runtime tie.

GitHub implementation repositories are collected separately and never counted as retrieved papers. Code/task/dataset links are source metadata, not relevance boosts. Coverage totals are query-result counts, not globally unique artifacts.

Recompute: `python -m benchmarks.academic_report benchmarks/reports/academic-samples`. Private snapshots are required; no paid API calls are made by reporting.

Upstream Voyage reranking returned HTTP 429 because its shared provider project was near a 12,000,000-token/minute limit. Bounded retries and attempts are retained. Failed model rows are never silently replaced with RRF. The original run eventually completed all global model rankings; the expanded run completed 13/34 global Voyage rankings (1/12 known-paper, 2/12 open-ended, 10/10 LitQA). Full-sample retrieval-only tables remain valid despite these model failures.

The original four-provider experiment compares Jev, D1, Voyage and packed ranker ensembles on identical frozen candidate sets. All 34 global ranking cases are complete after explicit bounded retries.
