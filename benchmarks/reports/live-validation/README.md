# Live contract and small ranking check

Run date: 2026-10-29. Three web and three scholar questions. No overall quality
claim is supported by this sample. See `manifest.json` for sources and sampling.

- `frozen.jsonl`: exact normalized inputs and qrels.
- `results.json`: saved rankings, pool hashes, resolved models, metrics, paired
  bootstrap intervals, costs and sanitized response metadata.
- `report.md`: readable comparison table. The scholar “single” row is an
  arbitrary corpus-pool ordering, not a search-provider baseline.

Reported costs, including the three initial contract checks: **$0.00184379**.
131 paid requests; conservative reservations totaled $0.8368368, below an
internal $1 ceiling and the user's $5 ceiling. The key was held in memory only.

To regenerate the report from saved results without API calls:

```sh
python -c "import json; from benchmarks.compare import report; print(report(json.load(open('benchmarks/reports/live-validation/results.json'))['records']))"
```

To repeat the paid ranking experiment with your own environment key:

```sh
python -m benchmarks.compare benchmarks/reports/live-validation/frozen.jsonl \
  --methods rrf jev decider voyage --shortlist 10 --candidate-tokens 300 \
  --ranking-concurrency 2 --output /tmp/ranking-repeat
```

The repeat command makes paid calls; this saved report can be read and rendered
without repeating it. The original one-off validation run additionally used a
restricted transport with retries disabled, at most 150 requests and 20 KB per
request, and a $1 aggregate conservative reservation limit (at least $0.005 per
request or 100 times the published highest input rate applied per request byte).
Output was free for these typed endpoints at the verified prices. That guard
was part of the validation harness, not the general comparison command.

Web qrels are released auto-heuristic judgments. LitSearch negatives are
unjudged distractors treated as nonrelevant; positive papers were deliberately
included. Recall@10 is unchanged across rankers because there are at most ten
ranked candidates. Treat nDCG differences and bootstrap intervals as diagnostic
only. No retrieval ensemble claim is validated by the constructed scholar pool.
