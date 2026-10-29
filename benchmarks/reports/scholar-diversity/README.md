# Scholar diversity experiment

Read the [fresh validation](validation-16/report.md) first. The tested Exa-preserving
blend did not beat Exa on the fresh sample and remains opt-in. Useful complementary
papers exist, but the tested selection rules did not preserve enough baseline
quality. The main `report.md` contains the earlier 36-development/20-held-out
candidate and score-band ablations. A later exploratory anchor repair was checked
on 16 additional, previously unused questions; those results were negative.

Artifact inventory:

- `manifest.json`: precommitted 56-query sample/split and retrieval costs.
- `frozen.jsonl`: normalized outputs, original ranks, failures and qrels.
- `gold-identifiers.json`: identifier metadata for known relevant papers; used
  for evaluation only. Papers are never inserted into retrieval outputs.
- `scored.jsonl`: one global Jev judgment per unique candidate, reused across
  candidate-selection policies because judgments are independent.
- `scoring.json`: actual model-stage reported spend, request count and snapshot hash.
- `comparison.json`: output-budgeted metrics and additional-paper retention.
- `validation-16/`: untouched query IDs, collected results, saved rankings,
  output-budgeted scores and the negative final validation.
- `budget.json`: reported charges plus a separate Parallel price estimate.

Rebuild the first comparison without API calls:

```sh
python -m benchmarks.diversity \
  --snapshot benchmarks/reports/scholar-diversity/frozen.jsonl \
  --scores benchmarks/reports/scholar-diversity/scored.jsonl \
  --manifest benchmarks/reports/scholar-diversity/manifest.json \
  --output /tmp/scholar-diversity-replay
```

Shared identities include DOI/identifier deduplication and compatible-title/year
links. Apply the same mapping to individual-provider baselines: otherwise richer
metadata can look like a new paper even when Exa already retrieved it. Only
annotated, genuinely non-Exa papers count toward the novelty metric.

Provider-balanced candidates and near-tie content novelty are separately
assessable. Neither guarantees that a distinct paper is relevant. There are no
final provider quotas. One model scores the combined set; it does not score each
provider separately. The original question remains the model input. Semantic
Scholar receives deterministic lexical cleanup; OpenAlex uses native semantic
search; Parallel receives a primary-paper retrieval objective.
