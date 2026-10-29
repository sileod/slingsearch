# Decision packing, evidence diagnostics, and provider-key audit

All samples and policies were recorded before the new scores. No default provider,
ranker, batch size, or evidence-selection policy was changed.

## Packing experiment: twelve unseen DORIS questions

The earlier sixteen DORIS question IDs were excluded. Each new question uses the
same fixed BM25 top20 shortlist from its full native pool, with identical original
question, document text, 400-estimated-token budget, and relevance rubric. Metrics
use the released full-pool aspect gains. These results must not be directly compared
with the earlier full-pool ranking table.

| Method | Requests/query | Input tokens/query | Median seconds | Reported USD/query | nDCG@5 |
|---|---:|---:|---:|---:|---:|
| Jev, one candidate/request | 20 | 16429 | 5.258 | 0.000690 | 0.7598 |
| Jev, five candidates/request | 4 | 9513 | 1.194 | 0.000400 | 0.7581 |
| Jev, relevance + two evidence checks | 20 | 18389 | 5.381 | 0.000772 | 0.7634 |
| Voyage | 1 | 7791 | 0.581 | 0.000156 | 0.7678 |

Packing reduces HTTP requests by 80%, reported input cost by 42.1%, and observed
median latency by about 4.4x. The nDCG@5 change is -0.0016, with paired-query
bootstrap 95% interval [-0.0211, 0.0121]. This is a small equivalence probe, not a
formal non-inferiority guarantee. Packed payloads move each document into its own
question's structured instructions; shared state contains only the original user
question. No candidate sees another candidate's text.

The two additional Noul questions ask whether the provided text contains useful
answer evidence and whether more source text would materially improve assessment.
Three judgments cost 11.9% more input than relevance alone, with similar latency.
They remain diagnostics; no fetch/stop decision, filtering, or quality improvement
from using those signals has been validated. Their presence does not change the
relevance sorting policy. Small relevance differences are not an evidence-policy win.

The API supplied no cached-token telemetry. The result demonstrates shared-input
amortization and packing, not a measured provider prompt-cache hit. Exact-response
replay is separately implemented in the private benchmark transport. The core
library has no persistent response cache. TypeSafe documents the underlying
[multiple-question pattern](https://docs.typesafe.ai/cookbooks/parallel_questions).

One Voyage request failed initially and succeeded on one explicit retry. Attempts
remain in local logs; no failed ranker was silently replaced with RRF. A loader
check was also corrected to inspect only aspect-paper pairs actually needed by
each selected query, avoiding irrelevant cross-query annotation conflicts. True
conflicting or missing evaluated judgments still fail validation.

See [packing summary and intervals](doris-summary.json).

## New scholar validation: eight previously unused questions

The eight PaperFindingBench semantic questions exclude the preceding twelve.
Exa, Parallel, OpenAlex, and Semantic Scholar each retrieve ten results. Metadata
enrichment retries bounded HTTP429 failures and deduplication occurs before the
historical June2025 date exclusion. Full raw results are retained locally. This
differs from the earlier metadata pipeline and should be disclosed when comparing
experiments. All eight provider searches succeed for each provider in this run.

| Method | Useful-paper P@5 |
|---|---:|
| Exa | 0.700 |
| Parallel | 0.325 |
| OpenAlex | 0.200 |
| Semantic Scholar | 0.175 |
| RRF blend | 0.625 |
| Jev blend, independent requests | 0.775 |
| Jev blend, packed5 | 0.750 |
| Voyage blend | 0.725 |
| Exa + Voyage | 0.775 |

Jev improves over raw Exa by 0.075 (bootstrap interval [0.025, 0.150]), but its mean
ties Exa+Voyage. Voyage blending is -0.050 versus Exa+Voyage, interval [-0.150,
0.050]. The independent Jev blend retains eight useful question-paper pairs absent
from raw Exa results. This confirms additional-paper contributions without proving
that blending always improves the best reranked individual source.

Packed Jev is -0.025 versus independent Jev, interval [-0.075, 0]. Its speed/cost
benefits therefore remain opt-in rather than an automatic default. Two transient
Voyage requests succeeded on one explicit retry each; all eight questions enter the
final paired comparison. Initial six-query common-success results are retained
locally for audit rather than silently discarded.

Judgments use the released question criteria and a distinct GPT-4o evaluator,
including the original known-positive override. This is a small bounded-output
adaptation with model judgments, not an official leaderboard score or independent
human validation. All methods use five sources / 1500 estimated output tokens.
See [scholar results and intervals](scholar/results.json).

## Web provider ablation: eight repeated questions, new live retrieval

These questions were seen in the earlier web holdout. The retrieval outputs and
independent judgments are new; this is not fresh-query validation. All five web
providers use the supplied keys and succeed on all eight questions. The original
information need is the evaluator criterion; no model generates search queries,
rubrics, or answers. This coarse usefulness judgment saturates and cannot establish
fine-grained evidence diversity or factual accuracy.

| Method | Useful-paper P@5 |
|---|---:|
| Exa | 0.875 |
| Parallel | 0.850 |
| Brave | 0.875 |
| Tavily | 0.800 |
| Firecrawl | 0.875 |
| RRF blend | 0.875 |
| Jev blend, independent requests | 0.975 |
| Jev blend, packed5 | 1.000 |
| Voyage blend | 0.975 |
| Exa + Voyage | 0.975 |
| Parallel + Voyage | 0.950 |

Blending plus Voyage ties Exa+Voyage on every query's P@5. Thus a gain over raw
providers cannot be attributed exclusively to provider diversity. The packed Jev
100% is forty model-judged slots on previously seen questions, not a general
accuracy claim. See [web results and intervals](web/results.json).

## Which supplied keys were used?

| Key | New search questions | Role |
|---|---:|---|
| EXA_API_KEY | 16 | Web and scholar retrieval |
| PARALLEL_API_KEY | 16 | Web and scholar retrieval |
| OPENALEX_API_KEY | 8 | Scholar semantic search |
| SEMANTIC_SCHOLAR_API_KEY | 8, plus metadata requests | Scholar lexical search and metadata |
| BRAVE_API_KEY | 8 | Web retrieval |
| TAVILY_API_KEY | 8 | Web retrieval |
| FIRECRAWL_API_KEY | 8 | Web search, not paid page scraping |
| OPENROUTER_API_KEY | All model experiments | Typed ranking/evidence judgments, reranking, independent benchmark relevance assessment |

All keys were held in process memory. Ordinary library calls still require the
caller's environment variables or explicit Config keys. No credentials, raw
provider evidence, gated source questions, or response caches are committed.

## Cost and reproduction

The persistent cross-process budget reserves money before requests, accounts for
in-flight work, and reuses exact cached responses on resume. Unreported Brave,
Tavily, Firecrawl, and OpenAlex charges retain conservative allowances rather than
being assumed free. Including earlier experiments and a $0.10 uncertainty allowance,
the session is conservatively accounted at approximately $3.82, below $5. This is
not an invoice; actual charges can be lower. See [budget ledger](api-budget.json).

Free local replay, requiring the retained private snapshots:

```sh
python -m benchmarks.decision_audit report
python -m benchmarks.decision_live report
```

Paid reproduction uses the corresponding `prepare` and `run` commands. Source
dataset access is explicit; hidden credential prompts avoid saving keys. New native
data and all raw requests/responses remain excluded from Git and package artifacts.
