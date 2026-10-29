# Reproducible evaluation

Install `pip install -e '.[bench,test]'`. Downloads are explicit; none occur
when importing the library or running ordinary searches. Keep a dataset
revision/run ID with each snapshot. Data licenses remain those of the datasets.

Published reports include aggregate results and experiment manifests. Downloaded
datasets, gated source files, raw provider snapshots, response caches, and local
per-query judgments are excluded from Git. Free replay commands require those
retained local artifacts; a fresh clone must explicitly acquire the source data
and run the evaluation first. No credentials are included.

## Frozen replay

A snapshot is JSONL with one query per line:

```json
{"id":"q1","question":"Question text","mode":"web","candidates":[{"id":"https://example.org/","title":"Title","url":"https://example.org/","passages":["Literal excerpt"],"sources":[{"provider":"exa","rank":1}]}],"qrels":{"https://example.org/":3},"provenance":"dataset revision and label origin","retrieval":{}}
```

Candidates retain original provider ranks and untrimmed normalized metadata.
Qrel keys can be canonical URLs, `doi:...` or `corpusid:...`. Replay pools are
fused and prepared once, then shared by all rankers with the same text budget.
The pool hash covers question, candidate IDs and text representations.

```sh
python -m benchmarks.compare benchmarks/fixtures/smoke.jsonl --methods rrf --output smoke
python -m benchmarks.compare frozen.jsonl --methods rrf jev decider voyage --output comparison
```

The synthetic smoke fixture only checks execution. Its scores say nothing
about real retrieval quality. Model comparisons make paid OpenRouter requests.
Ranker failures are failed measurements; there is no benchmark fallback.

The JSON records include nDCG@5/10, Recall@10/20 and reciprocal rank@10 via
ir-measures, latency, reported cost, token estimates, success, result IDs,
provider-exclusive relevant-document contributions, and provenance. Unjudged
documents count as nonrelevant. Recall is relative to available qrels, which
can be incomplete. Models' raw scores are never compared across backends.

The Markdown table shows separate web and scholar views. Best single means
the provider with highest aggregate nDCG@5 on this evaluation set, an optimistic
baseline rather than a per-query oracle. Model metrics use the identical
shortlist; single providers retain their complete released rankings. Paired
bootstrap comparisons against RRF use common successful query IDs, 2,000
resamples and seed 0. Examine failure rates for survivorship bias.

Latency and cost scopes are explicit: offline model rows measure ranking only;
single rows use retrieval telemetry when supplied. Unknown cost is `NA`, never
an assumed zero. JSON adds median/p95 latency and coverage of reported costs.
For end-to-end costs add retrieval snapshot costs to ranking costs; do not
interpret ranking-only costs as total search costs.

## Native dataset loaders

Load files you have explicitly downloaded, then save normalized queries:

```python
import json
from dataclasses import asdict
from benchmarks.datasets import needle, websearch_kit, litsearch, sparbench

queries = needle.load("runs/PINNED-RUN/rbp.json")
# Or:
queries = websearch_kit.load(
    "retrieved_results.csv", "pooled_judgments_primary.csv", "queries.jsonl",
    repetition=1, variant="primary", runs_path="runs.csv",
)
# Academic queries without frozen target-provider outputs:
queries = litsearch.load("litsearch-query.json")
# Selected question-paper pools, not a broad retrieval corpus:
queries = sparbench.load("sparbench.json")

with open("frozen.jsonl", "w") as f:
    for q in queries:
        f.write(json.dumps(asdict(q)) + "\n")
```

Loaders also accept JSONL for native LitSearch/SPARBench rows. Hugging Face
downloads are opt-in via `litsearch.load(revision="COMMIT")` and
`sparbench.load(revision="COMMIT")`; the revision is mandatory for downloads.

- [NEEDLE](https://huggingface.co/datasets/keenable-ai/needle-results): the
  loader supports engine reports with `engines.*.per_query[].results` and
  per-document `rating` values. Conflicting URL judgments require adjudication.
  Reports containing only answer-hit ranks are rejected as document qrels.
  NEEDLE is maintained by a search provider; complement it with independent
  data and preserve its label provenance.
- [Web Search Benchmark Kit](https://github.com/Microck/websearch-bench): load
  the released CSV results and one pooled judgment file, plus query catalog.
  Choose repetition and query variant; neither is silently pooled. Released
  judgments can be heuristic or AI-generated: select and disclose the source.
- [LitSearch](https://huggingface.co/datasets/princeton-nlp/LitSearch): native
  rows contain `query` and relevant `corpusids`. This creates known-relevant
  qrels, **no frozen retrieval results**. Collect live outputs before replay.
- [SPARBench](https://huggingface.co/datasets/XiaofengAlg/SPARBench): native
  rows contain `query` and `answers` (paperID, title, abstract, etc.). Released
  Qwen scores are not automatically treated as human relevance judgments.
  Default loading provides unlabeled candidate pools. Supply an explicit
  integer `grade_field` for reviewed annotations; continuous model scores
  are rejected. Treat this as supplementary judgment evaluation.

## DORIS-MAE and PaperFindingBench

The DORIS loader reads the [official Zenodo v1 JSON](https://zenodo.org/records/8299749)
with `ijson`, retaining only sampled candidate abstracts and their released aspect
judgments. Download the JSON explicitly (about 1 GB); embeddings are unnecessary.
The data has its own CC-BY-NC license and is not included in the package.

```sh
python -m benchmarks.scholar_eval prepare --native DORIS-MAE_dataset_v1.json \
  --out runs/doris --size 16
# Paid; requires OPENROUTER_API_KEY. Reuses completed query/model pairs and responses.
python -m benchmarks.scholar_eval run --out runs/doris
# Free replay of saved rankings and labels:
python -m benchmarks.scholar_eval report --out runs/doris
```

The sample seed and upstream 60-query sampling frame are saved before ranking.
All methods see the full native candidate pool and identical 400-estimated-token
document text. Baselines are BM25 and one seeded random order. Jev, Decider, and
Voyage use the library adapters. No provider retrieval or RRF ranking is invented
for a dataset without target-provider rankings. This test cannot establish that
blending beats Exa.

Linear nDCG uses the sum of released AnnoGPT aspect judgments, matching upstream
normal nDCG; these are not independent human labels. Recall uses the upstream
mean-aspect-score threshold of 1 and excludes queries with no qualifying papers.
Aspect coverage measures the sum of best retrieved judgments per aspect relative
to the full pool. Treat it as a diagnostic alongside relevance, since partial
matches can cover aspects without meeting the complete question. JSON includes
paired query bootstrap intervals and actual 1500-token output metrics.

[PaperFindingBench](https://huggingface.co/datasets/allenai/asta-bench) requires
dataset access. `benchmarks.datasets.paperfinding.load(path)` reads the official
validation/test JSON and preserves the scoring criteria. Semantic known-positive
sets remain **non-exhaustive**, so the loader does not convert them into ordinary
exhaustive qrels. Semantic evaluation requires judging additional retrieved papers
against the released criteria using a separate evaluator. The official scorer's
recall denominator is estimated, not a complete census of relevant papers.
Use the published cutoff and verify any scorer fixes before comparing official
scores. Missing access is a pending evaluation, never a zero score or a substitute
sample described as the official benchmark.

The [sampled scholar evaluation](reports/scholar-benchmarks/README.md) includes
completed DORIS-MAE and PaperFindingBench experiments, query bootstrap intervals,
provider-failure diagnostics, and cost accounting. To reproduce PaperFindingBench,
download the official validation JSON, normalizer reference, and scorer prompt
source into the explicit `--raw` directory. The source hashes and pinned dataset
revision are recorded in the manifest. Then run:

```sh
python -m benchmarks.paperfinding_eval prepare
python -m benchmarks.paperfinding_eval collect-rank  # paid; hidden credential prompt
python -m benchmarks.paperfinding_eval judge         # paid; separate relevance evaluator
python -m benchmarks.paperfinding_eval report        # free saved-judgment replay
```

The independent evaluator uses the upstream GPT-4o model and criteria through
OpenRouter exclusively for relevance assessment. It does not produce search queries
or answers, and adds no chat-model wrapper to the core ranking API. The runner
normalizes case and underscores in criterion names and fails explicitly on missing
criteria. Its small fixed-k adaptation is reported separately from official
leaderboard scores. Changing judge, source text, or criteria requires a new run.

## Live collection and paper matching

```sh
python -m benchmarks.live litsearch-query.json --litsearch \
  --providers openalex semantic_scholar exa --corpus corpus-map.json \
  --output frozen.jsonl
python -m benchmarks.compare frozen.jsonl --methods rrf jev decider voyage
```

For web questions use the normalized query JSONL without `--litsearch`.
Collection calls each provider once per query and saves normalized results,
original ranks and retrieval status/timing/cost. It does not rerank or trim
abstracts before saving. Run single-provider baselines from this same snapshot.

`corpus-map.json` maps `corpusid:ID` to paper metadata (`doi`, `identifiers`,
`title`, `year`). Matching prefers exact DOI/identifier, then a unique exact
normalized title with matching publication year. Inspect title matches manually
before publishing benchmark results. Semantic Scholar's returned CorpusId can
match LitSearch directly. Unknown matches stay unmatched rather than using
fuzzy title guesses. The corpus mapping must use the same LitSearch revision.

## Passage evaluation

```sh
python -m benchmarks.passages frozen.jsonl --size 100 --output passage-assessment.csv
```

This creates a deterministic seed-0 manual-assessment sample with question,
source URL, literal passage, output tokens and empty fields for usefulness,
verification at the URL, and notes. Human assessors fill those fields; keep
this evaluation separate from document relevance scores. No automatic passage
quality improvements are claimed.

## Exa-preserving fusion and held-out comparison

`benchmarks.anchor` contains an experimental rule that preserves Exa's first
result and scores the remainder by Exa rank plus supplementary rank support.
It is outside the product defaults. A small parameter grid is selected only on
development query labels; held-out labels cannot select parameters.

```sh
python -m benchmarks.anchor_compare \
  --results retrieved_results.csv --judgments pooled_judgments_primary.csv \
  --queries queries.jsonl --runs runs.csv \
  --development-manifest benchmarks/reports/sampled-comparison/manifest.json \
  --exclude-ids KI02 KI03 --output /tmp/anchor-holdout
```

The saved [query-held-out report](reports/anchor-holdout/report.md) excludes both
the 20 exposed development questions and two earlier smoke-check questions.
It compares 57 other eligible questions without paid requests. These released
heuristic labels support a within-benchmark finding, not independent human
validation. No provider weights or product defaults are chosen from the holdout.

Individual-provider baselines use the same globally resolved document identities
as the ensemble. Otherwise a second provider's DOI/CorpusId metadata can make an
already-retrieved paper look like a new retrieval hit. This shared normalization
can link versions through compatible title/year matching; inspect ambiguous links
before publishing retrieval-quality claims.
