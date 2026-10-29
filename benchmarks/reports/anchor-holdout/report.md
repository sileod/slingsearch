# Exa-preserving fusion: development and query-held-out evaluation

Subsequent best-individual audit: Parallel scores 0.7356 nDCG@5 on the same
57-question holdout, above Exa's 0.5798. RRF trio's advantage over Parallel is
only 0.0180, with paired bootstrap 95% interval [-0.0273, 0.0617]. The large
Exa-relative improvement below must not be described as a demonstrated win
over the strongest individual provider. This audit reused frozen results and
labels without paid API calls.

No paid requests. Parameters selected on the previously exposed 20 queries; earlier smoke-check questions KI02/KI03 are also excluded from holdout. Heuristic relevance labels limit the conclusion.

Selected rule: preserve Exa top 1; supplementary rank-support weight 2.

| Split | Queries | Exa nDCG@5 | RRF trio | Selected anchor |
|---|---:|---:|---:|---:|
| development | 20 | 0.6976 | 0.7314 | 0.7366 |
| holdout | 57 | 0.5798 | 0.7536 | 0.6995 |

Paired holdout intervals vs Exa:

{
  "rrf_trio": {
    "mean_delta": 0.17381681489491707,
    "ci95": [
      0.113574433602822,
      0.2378346685042207
    ],
    "queries": 57,
    "seed": 0,
    "samples": 2000
  },
  "anchor_selected_on_dev": {
    "mean_delta": 0.11974292663586658,
    "ci95": [
      0.07911559700338043,
      0.16042569142378504
    ],
    "queries": 57,
    "seed": 0,
    "samples": 2000
  }
}
