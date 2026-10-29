# Sampled comparison artifacts

Read `report.md` for results and limitations; `manifest.json` records sampling,
source hashes and ID-matching corrections. No credentials are stored here.

- `frozen.jsonl`: corrected normalized results with exact gold identifier matches.
- `frozen-before-id-fix.jsonl`: original inputs for audit. Model candidate IDs,
  texts and every provider ordering were verified identical after the fix.
- `results.json`: saved method rankings, metrics, timing, reported ranking cost,
  operational status and paired bootstrap comparisons.
- `default-web-trio.json`: free RRF comparison using Exa, Parallel and Brave only.
- `summary.json`: per-method aggregate metrics, including blending vs Exa CIs.
- `gold-identifiers.json`: known relevant papers' metadata, fetched by CorpusId.

Total known reported spend, including prior validation, is $0.083066482.
This run made 1,108 ranking requests ($0.017222692 reported), plus 24 live scholar
retrieval attempts ($0.064 reported). No paid calls were needed for the ID fix.

Regenerate the ordinary comparison table without API calls:

```sh
python -c "import json; from benchmarks.compare import report; print(report(json.load(open('benchmarks/reports/sampled-comparison/results.json'))['records']))"
```

Repeat the paid comparison on the frozen inputs:

```sh
python -m benchmarks.compare benchmarks/reports/sampled-comparison/frozen.jsonl \
  --methods rrf jev decider voyage --shortlist 20 --candidate-tokens 300 \
  --ranking-concurrency 5 --output /tmp/sampled-repeat
```

The separate `python -m benchmarks.run_sample --help` command can select native
release inputs and collect fresh literature results. It uses environment keys
(or one stdin JSON dictionary), category-balanced seed-42 web sampling,
seed-42 scholar sampling, and a restricted transport reserving at most $3.50 for
ranking with retries disabled. Retrieval costs are separate. Use the default
small sample sizes; the spend reservation is specific to the verified model
prices and is not an account-level billing limit.
