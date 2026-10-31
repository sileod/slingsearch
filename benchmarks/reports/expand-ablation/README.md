# Expand add-one-provider ablation

Keep Expand opt-in: this experiment cannot establish a reliable gain for the shipped Jev default.

16 newly sampled questions: 8 FreshQA stable true-premise web questions and 8 SAGE questions (one per task/domain stratum). Seed 20261031; 208 previously used local questions excluded. All providers requested 10 results; shortlist 20; candidate text 400 tokens; final maximum 5 sources and 1,500 tokens.

## Web: actual displayed evidence

Independent free Nemotron judgment, blinded to provider/method/rank. Grade 0 unrelated or contradictory, 1 topical only, 2 partial supporting evidence, 3 direct answer evidence. Mean grade is among returned sources; empty output scores zero. These are model estimates, not human judgments.

| Ranking / retrieval | Queries | Mean grade (0–3) | Useful precision | Direct-evidence query coverage | Useful sources / 1k tokens | Returned |
|---|---:|---:|---:|---:|---:|---:|
| rrf:default | 8 | 1.375 | 0.400 | 0.625 | 1.334 | 5.00 |
| rrf:default+expand | 8 | 1.675 | 0.500 | 0.750 | 1.676 | 5.00 |
| rrf:expand | 8 | 1.750 | 0.500 | 0.875 | 1.761 | 5.00 |
| span_free:default | 8 | 1.850 | 0.625 | 0.875 | 1.994 | 4.62 |
| span_free:default+expand | 8 | 1.942 | 0.600 | 1.000 | 1.933 | 4.75 |
| span_free:expand | 8 | 1.762 | 0.550 | 0.750 | 1.729 | 4.62 |

Paired differences are **default+Expand minus default**:

- rrf:web:grade: +0.300, query-bootstrap 95% CI [+0.025, +0.575], n=8.
- rrf:web:useful_precision: +0.100, query-bootstrap 95% CI [-0.100, +0.300], n=8.
- rrf:web:direct_query: +0.125, query-bootstrap 95% CI [+0.000, +0.375], n=8.
- rrf:web:useful_per_1000_tokens: +0.342, query-bootstrap 95% CI [-0.325, +1.007], n=8.
- span_free:web:grade: +0.092, query-bootstrap 95% CI [-0.250, +0.442], n=8.
- span_free:web:useful_precision: -0.025, query-bootstrap 95% CI [-0.150, +0.100], n=8.
- span_free:web:direct_query: +0.125, query-bootstrap 95% CI [+0.000, +0.375], n=8.
- span_free:web:useful_per_1000_tokens: -0.061, query-bootstrap 95% CI [-0.500, +0.355], n=8.

## Scholar: known reference papers

| Ranking / retrieval / task | Queries | nDCG@5 | Hit@5 | WeightedRecall@5 | Recall@10 after ranking/cutoff | Returned |
|---|---:|---:|---:|---:|---:|---:|
| rrf:default:short | 4 | 0.000 | 0.000 | 0.000 | 0.000 | 4.75 |
| rrf:default:open | 4 | 0.092 | 0.500 | 0.098 | 0.142 | 5.00 |
| rrf:default+expand:short | 4 | 0.000 | 0.000 | 0.000 | 0.000 | 4.75 |
| rrf:default+expand:open | 4 | 0.055 | 0.250 | 0.062 | 0.142 | 5.00 |
| rrf:expand:short | 4 | 0.000 | 0.000 | 0.000 | 0.000 | 5.00 |
| rrf:expand:open | 4 | 0.000 | 0.000 | 0.000 | 0.000 | 5.00 |
| span_free:default:short | 4 | 0.000 | 0.000 | 0.000 | 0.000 | 0.75 |
| span_free:default:open | 4 | 0.055 | 0.250 | 0.062 | 0.042 | 4.25 |
| span_free:default+expand:short | 4 | 0.000 | 0.000 | 0.000 | 0.000 | 0.25 |
| span_free:default+expand:open | 4 | 0.000 | 0.000 | 0.000 | 0.042 | 4.25 |
| span_free:expand:short | 4 | 0.000 | 0.000 | 0.000 | 0.000 | 0.00 |
| span_free:expand:open | 4 | 0.000 | 0.000 | 0.000 | 0.000 | 3.75 |

## Retrieval and operational scope

- web: provider success {'exa': 8, 'parallel': 8, 'brave': 8, 'expand': 8}; retrieval p50 seconds {'default': 2.271100535057485, 'default+expand': 2.271100535057485, 'expand': 1.1533769518136978}.
- scholar: provider success {'exa': 8, 'parallel': 8, 'openalex': 8, 'semantic_scholar': 6, 'serpapi': 8, 'expand': 8}; retrieval p50 seconds {'default': 3.985817465931177, 'default+expand': 3.985817465931177, 'expand': 1.3197042234241962}.
- Expand contributed 0 additional known-positive query-paper pairs; unresolved reference IDs: 0.
- Judge spot-check: {"questions": 8, "exact_grade_agreement": 0.5, "useful_binary_agreement": 1, "scope": "Assistant audit of one randomly selected blinded excerpt per question, without viewing judge labels first; not human validation"}
- Incremental Expand search cost: documented $1.50 per 1,000 calls; provider-reported USD cost unavailable. Free Span and independent judge report zero model cost.

## Limitations

- Eight web questions and eight scholar questions: exploratory, low statistical power
- Held out from 208 prior local experiment questions, not guaranteed held out from provider training
- FreshQA stable-fact adaptation evaluates evidence excerpts, not FreshQA answer generation
- Web labels come from an independent free-model blinded audit, not humans or full-page verification
- The eight-item assistant spot-check agreed on useful/not-useful labels but only half of exact ordinal grades; at least one rationale inferred a game name beyond the displayed excerpt. Fine-grained grade effects should be treated cautiously
- Span-01 Lite binary evidence scores are not numerically equivalent to Jev ordinal rubric scores
- Jev is blocked by the existing key cap. This does not validate the shipped Jev default
- SAGE reference labels are incomplete; known-paper coverage does not measure all academic relevance
- All arms share frozen base retrieval and identical limits; Expand adds one retrieval call
- Source identity enrichment is used for scoring only; no target abstracts or answers enter ranking
- Latency is retrieval-only (maximum constituent provider latency); shared free-model scoring is amortized and not a production latency comparison

Reproduce with retained private snapshots: `python -m benchmarks.expand_ablation rrf`, `python -m benchmarks.expand_ablation free`, `python -m benchmarks.expand_ablation blind`, `python -m benchmarks.expand_judge`, `python -m benchmarks.expand_report`. Fresh collection requires the pinned FreshQA CSV and SAGE release plus configured credentials. Paid calls share the persistent session budget.

Sources: [FreshQA](https://github.com/freshllms/freshqa), [SAGE](https://github.com/HughieHu/Sage), [free Span](https://openrouter.ai/respan/span-01-lite:free), [Expand](https://expand.ai/docs/search).
