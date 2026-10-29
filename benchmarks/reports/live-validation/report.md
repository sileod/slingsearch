# Live ranking validation — six questions

Small execution check, not independent validation of quality. Web labels are released heuristic judgments. Scholar papers use a constructed candidate pool; single-pool baseline is not a search provider. Three questions per mode give unstable confidence intervals. Ranking rows measure ranking-only latency and cost.

Offline replay: model rows measure ranking latency/cost only; single rows use released retrieval telemetry.
Unknown costs/timings are shown as NA. Unjudged documents count as nonrelevant. Best single is selected by aggregate nDCG@5 on this evaluation set (optimistic).

web

| Method | nDCG@5 | Recall@10 | Latency p50 (s) | Cost/1000 queries | Success |
|---|---:|---:|---:|---:|---:|
| Best single (kagi_session) | 0.8319 | 0.2762 | NA | NA | 1.0000 |
| rrf | 0.8360 | 0.2762 | 0.0005 | 0.0000 | 1.0000 |
| jev | 0.7631 | 0.2762 | 1.1120 | 0.2217 | 1.0000 |
| decider | 0.8049 | 0.2762 | 1.3083 | 0.0596 | 1.0000 |
| voyage | 0.6708 | 0.2762 | 0.2431 | 0.0290 | 1.0000 |

scholar

| Method | nDCG@5 | Recall@10 | Latency p50 (s) | Cost/1000 queries | Success |
|---|---:|---:|---:|---:|---:|
| Best single (fixed_corpus_pool) | 0.3916 | 1.0000 | NA | NA | 1.0000 |
| rrf | 0.3916 | 1.0000 | 0.0002 | 0.0000 | 1.0000 |
| jev | 1.0000 | 1.0000 | 0.9581 | 0.2123 | 1.0000 |
| decider | 1.0000 | 1.0000 | 1.2697 | 0.0543 | 1.0000 |
| voyage | 1.0000 | 1.0000 | 0.2212 | 0.0239 | 1.0000 |

Reported request cost: $0.001843790

Conservative reservation: $0.836836800 (< $1; user ceiling $5).
