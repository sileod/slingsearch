# Synthetic execution check

These are synthetic fixture scores, not retrieval-quality measurements. Model backends were not run.

Reproduce: `python -m benchmarks.compare benchmarks/fixtures/smoke.jsonl --methods rrf --output /tmp/evidence-smoke`

Offline replay: model rows measure ranking latency/cost only; single rows use released retrieval telemetry.
Unknown costs/timings are shown as NA. Unjudged documents count as nonrelevant. Best single is selected by aggregate nDCG@5 on this evaluation set (optimistic).

web

| Method | nDCG@5 | Recall@10 | Latency p50 (s) | Cost/1000 queries | Success |
|---|---:|---:|---:|---:|---:|
| Best single (provider_a) | 1.0000 | 1.0000 | NA | NA | 1.0000 |
| rrf | 1.0000 | 1.0000 | 0.0001 | 0.0000 | 1.0000 |
| jev | NA | NA | NA | NA | 0.0000 |
| decider | NA | NA | NA | NA | 0.0000 |
| voyage | NA | NA | NA | NA | 0.0000 |

scholar

| Method | nDCG@5 | Recall@10 | Latency p50 (s) | Cost/1000 queries | Success |
|---|---:|---:|---:|---:|---:|
| Best single (provider_a) | 1.0000 | 1.0000 | NA | NA | 1.0000 |
| rrf | 1.0000 | 1.0000 | 0.0001 | 0.0000 | 1.0000 |
| jev | NA | NA | NA | NA | 0.0000 |
| decider | NA | NA | NA | NA | 0.0000 |
| voyage | NA | NA | NA | NA | 0.0000 |
