# SlingSearch

Small Python library for multi-provider evidence retrieval. Ask a question,
search providers concurrently, merge duplicates, fuse ranks, and optionally
rank one global shortlist. Returns source text and citations, never an answer.

The hypothesis is measurable: an ensemble plus a global relevance model might
produce better evidence per token than a single search provider. It might not.
The included replay tools compare both without changing retrieval outputs.

Requires Python 3.11+. From this checkout:

```sh
pip install -e '.[test,bench]'
pytest -q
```

## Web example

```python
from slingsearch import search

response = search(
    "How does Python cancellation propagate through an asyncio TaskGroup?",
    providers=["exa", "parallel", "brave"],
    ranker="rrf",  # free deterministic ranking
)
print(response.to_markdown())
# response.to_json(indent=2) or response.to_dict() for structured output
```

## Scholar example

```python
from slingsearch import asearch

response = await asearch(
    "What are the limitations of diffusion models for time-series forecasting?",
    mode="scholar",
    ranker="jev",  # default; explicitly falls back to RRF if unavailable
    year=(2020, 2026),
    max_results=5,
    token_budget=1500,
)
print(response.to_markdown())
```

Scholar defaults are OpenAlex, Semantic Scholar, Exa. Web defaults are Exa,
Parallel, Brave. Override either with `providers=[...]` or `Config`.
Scholar metadata includes DOI, identifiers, authors, year, venue and citation
count when supplied by providers. OpenAlex uses its native `search.semantic`
endpoint for natural-language questions; sending the entire question to keyword
AND search was a retrieval mismatch in the earlier experiment. Citation count never affects ranking.
Abstracts are used for passage selection and ranking; the compact response
includes an abstract excerpt only when it fits. Full normalized abstracts are
retained in live benchmark snapshots.

## Benchmarking example

```sh
# Deterministic, synthetic smoke comparison (not evidence of search quality):
python -m benchmarks.compare benchmarks/fixtures/smoke.jsonl --methods rrf --output smoke
# Paid ranking comparison on exactly the same frozen shortlist:
python -m benchmarks.compare frozen.jsonl --methods rrf jev decider voyage --output comparison
```

The command writes a Markdown table and JSON with per-query metrics, rankings,
candidate-pool hashes, actual model identifiers, operational measurements and
paired bootstrap intervals (2,000 resamples, seed 0). Web and scholar are
reported separately. Missing keys or failed models produce failed rows with
`NA` metrics, never a mislabeled RRF result. The comparison does not imply any
backend is better before measurements exist.

See [benchmark instructions](benchmarks/README.md) for native dataset loading,
live snapshots, scholarly ID matching and manual passage assessment.

## Credentials and configuration

Set only the keys you use in your shell or process environment:

```text
EXA_API_KEY
PARALLEL_API_KEY
BRAVE_API_KEY
TAVILY_API_KEY
FIRECRAWL_API_KEY
SEMANTIC_SCHOLAR_API_KEY
OPENALEX_API_KEY
OPENROUTER_API_KEY
```

Missing keys skip providers requiring authentication and produce response
warnings. OpenAlex and Semantic Scholar can run keyless with lower limits.
A free [OpenAlex key](https://help.openalex.org/api/authentication/) increases
its available budget. Keys are sent in headers, not saved in snapshots or
exception messages. Config representations omit the keys. The library does
not log requests; avoid enabling HTTP header logging in your application.

```python
from slingsearch import Client, Config

config = Config(
    web_providers=("exa", "parallel"),
    timeout=20,
    ranking_concurrency=5,
    candidate_tokens=400,
)
async with Client(config) as client:
    first = await client.asearch("What is reciprocal rank fusion?", ranker="rrf")
    second = await client.asearch("When does rank fusion fail?", ranker="voyage")
```

Use `with Client(...)` and `client.search(...)` for synchronous connection
reuse. Use one client in one event loop; do not mix sync and async contexts.
`search()` detects running event loops and directs notebooks to `asearch()`.
No global configuration file is required. Explicit keys can be supplied via
`Config(api_keys={"exa": "...", "openrouter": "..."})`.

## Ranking and evidence budgets

Pipeline: retrieval → normalization/deduplication → RRF → bounded local passage
selection → global ranking → bounded presentation. Modules are independent.

| Ranker | OpenRouter model | Contract |
|---|---|---|
| `rrf` | none | sum of `1 / (60 + provider_rank)` |
| `jev` (default) | `~typesafe/jev-latest` | typed Decisions score |
| `d1` | `~liquid/d1-ultra-latest` | typed Decisions score |
| `ensemble` (experimental) | configured members | equal-weight RRF of model rankings |
| `decider` | `perplexity/pplx-decider-v1.2` | typed Decisions score |
| `voyage` | `voyageai/rerank-3-lite` | rerank document indices |

The Jev default follows OpenRouter’s rolling latest alias (currently Jev 1.14).
Historical benchmark records retain the pinned model actually evaluated; their
scores do not automatically apply to future alias targets. Responses record the
requested model and any resolved model identifiers reported by OpenRouter.

Decisions use `POST /api/alpha/decisions`, a named score question, and five
ordered criteria. The API's zero-based score is shifted to the public 1–5
rubric. Candidates are judged independently with bounded concurrency. Voyage
uses `POST /api/v1/rerank`; incomplete/duplicate mappings fail validation.
Both receive the same title and selected passages, bounded by
`Config.candidate_tokens`. Scores are only meaningful within their backend;
ties use RRF. No OpenRouter calls generate queries, search, extract passages,
embed text or synthesize answers.

`ranker="d1"` uses the same bounded candidate representations and Decisions
adapter as Jev. `ranker="ensemble"` runs Jev, D1 and Voyage concurrently on the
same shortlist, then fuses **ranks** with RRF (`k=60`), never raw scores. Configure
members with `Config(ensemble_rankers=("d1", "voyage"))`. Ensemble model scores
are fusion scores, not relevance probabilities. Member costs add; member usage
and failures are exposed in `usage.ensemble_members`. If any member fails, the
whole stage falls back explicitly to retrieval RRF. Ensembling is experimental;
complementary model errors and quality improvements are not assumed.

Decision calls can optionally pack independent candidate questions:

```python
config = Config(decision_batch_size=5, decision_evidence_checks=True)
```

The default batch size remains one. Packing shares the original user question;
each typed question contains only its own bounded document. Candidates never see
other candidate texts. `decision_evidence_checks=True` also asks whether the text
contains useful answer evidence and whether more source text would help assess
it. These probabilities appear in `response.usage["decision_evidence"]` alongside
available relevance confidence/probabilities. They are diagnostics: they do not
automatically fetch pages, filter sources, or change the relevance ranking.

`usage.ranking_requests` and `usage.decision_questions` distinguish HTTP calls
from typed judgments. `cached_tokens` and `cache_write_tokens` are included only
when the upstream API reports them; absence means unknown, not zero. Shared-input
batching and exact-response replay are distinct from verified provider prompt-cache
hits. See the [decision packing evaluation](benchmarks/reports/decision-audit/README.md).

Partial decision failures settle all in-flight requests and discard partial
model scores before RRF fallback. The response records the actual backend,
requested backend/model, resolved model when available, usage, reported costs,
per-stage timings and warnings. Costs may remain unknown and exclude costs
not reported by upstream APIs. Pricing is not hard-coded.

Defaults: 10 results per provider, a 20-document shortlist (up to 50), 5 final
results, 2 passages per result, 180 estimated tokens per passage, and a 1,500
estimated-token output budget. `per_provider`, `shortlist`,
`passages_per_result`, `max_results` and `token_budget` are search arguments.
Provider-side caps can return fewer results than requested.

Tokens use a conservative UTF-8 byte/3 estimate, **not a model tokenizer**.
The output budget applies to the larger of compact result-array JSON and
citation Markdown, including titles, URLs, identifiers and provenance.
Response-envelope diagnostics and pretty-print whitespace are outside that
budget. Passages are shortened by taking literal prefixes; results are dropped
if mandatory metadata does not fit. Very small budgets can return no results.
`usage.output_tokens` reports this same estimate. Ranking text budgets are
separate from presentation budgets.

Passages remain literal source excerpts. Local term matching chooses among
provider excerpts or scholarly abstracts. `fetch_pages=True` optionally fetches
shortlisted web pages with insufficient text; install
`pip install 'slingsearch[extract]'`. Extraction is size-limited, uses
trafilatura, and failures leave available snippets intact. There is no crawler
or model-based passage extraction. Provider snippets can themselves contain
upstream processing; the library does not verify a snippet against its page.

DOIs, canonical URLs, shared scholarly identifiers, then normalized scholarly
titles with compatible years determine duplicates. ArXiv version IDs are
linked through their common paper ID. Provenance retains every original rank;
RRF counts each provider's best rank once. URL normalization retains paths,
trailing slashes and nontracking query parameters. Scholar `year=2025` or
`year=(2020, 2026)` uses native filters where available and a final metadata
filter; papers without a known year are excluded with a warning.

## Useful diversity in scholar search

To preserve candidate coverage from independent providers while keeping a strong
retrieval result visible:

```python
response = search(
    "Which papers study multilingual semantic parsing without full parsers?",
    mode="scholar",
    providers=["exa", "openalex", "semantic_scholar", "parallel"],
    ranker="jev",
    diversity=True,
    anchor_provider="exa",
    shortlist=20,
    max_results=5,
    token_budget=1500,
)
```

`diversity=True` reserves shortlist coverage by original provider ranks before
one global model ranking. After scoring, local text-overlap and provenance
novelty break narrow within-query relevance bands. It does not reserve final
slots for weak providers or call a second model. `anchor_provider="exa"`
explicitly preserves that provider's best available shortlisted result at #1;
model scores are retained unchanged, and remaining sources keep the global
ordering. Both settings appear in response usage metadata and are opt-in. The
[fresh 16-query validation](benchmarks/reports/scholar-diversity/validation-16/report.md)
did not beat Exa overall: the policy recovered one additional relevant paper but
lost other coverage. Treat these as experimental controls, not a demonstrated
quality improvement.

Semantic Scholar gets a deterministic lexical version of the question, with
conversational filler removed and hyphens split; scientific acronyms/dataset
names are retained. The ranking model always sees the original full question.
Parallel scholar mode asks for primary papers and abstracts. OpenAlex uses
native semantic search. Scholarly APIs retry HTTP 429 once with a bounded wait;
failures remain explicit if the retry fails.

Presentation reserves citation metadata first and shares excerpt space across
the selected papers. This helps complementary papers remain visible within the
same 1,500-token budget. Useful diversity is measured as **known relevant papers
not retrieved by Exa that survive into the final five**, not just the number of
provider names. See the [scholar diversity experiment](benchmarks/reports/scholar-diversity/report.md).

## CLI and optional MCP

```sh
slingsearch 'What limits diffusion forecasting?' --mode scholar --ranker rrf --format json
pip install 'slingsearch[mcp]'
slingsearch-mcp  # stdio; exposes search(question, mode, ranker, max_results, token_budget)
```

The CLI uses standard-library argparse. The optional MCP server delegates to
the same API; it adds no retrieval or ranking behavior.

## Validation status

Tests use fixed HTTP responses and cover all seven provider adapters, ranking
contracts, failures, pooling, deduplication, budgets and all four dataset
loaders. No paid API requests or model-quality claims are part of these tests.
A live one-result smoke check on 2026-10-29 succeeded for all seven retrieval
providers. Semantic Scholar returned no abstract for its first result. Live
Jev, Decider and Voyage contract checks also passed on 2026-10-29. A six-question
comparison and frozen inputs are in [the validation report](benchmarks/reports/live-validation/report.md);
this small check does not establish ranking quality. A larger [sampled comparison](benchmarks/reports/sampled-comparison/report.md)
now covers 20 web and 8 live scholar queries. Blending helped web in the sample
but hurt scholar; Jev is not established as the best reranker. A further
[57-question held-out web comparison](benchmarks/reports/anchor-holdout/report.md)
finds gains for both the default RRF trio and an experimental Exa-preserving rule.
[Scholar ablations](benchmarks/reports/semantic-scholar-ablation/report.md)
separate native OpenAlex semantic retrieval from Exa-only reranking; the small
scholar gains remain inconclusive. MCP requires its optional dependency.

Contract references: [OpenRouter OpenAPI](https://openrouter.ai/openapi.json),
[rerank API](https://openrouter.ai/docs/api/api-reference/rerank/create-rerank),
[Parallel Search](https://parallel.ai/products/search),
[Brave web search](https://api-dashboard.search.brave.com/app/documentation/web-search),
[Semantic Scholar](https://api.semanticscholar.org/api-docs/snippets),
[Firecrawl Search](https://docs.firecrawl.dev/api-reference/endpoint/search),
[Tavily Search](https://help.tavily.com/articles/4840311948-tavily-search-api),
[Exa](https://docs.exa.ai/reference/search).

## Next academic evaluations

[SAGE](https://github.com/HughieHu/Sage) and
[AstaBench LitQA2-FullText-Search](https://allenai.org/asta/bench) are priority
retrieval evaluations. Prepare pinned samples without search/model charges:

```sh
python -m benchmarks.academic_samples --output benchmarks/reports/academic-samples
```

The default sample has 24 SAGE questions (three per task/domain stratum) plus
the ten official LitQA validation questions. Query text is taken verbatim; reference
answers, target titles and generation plans are not appended. Downloaded data and per-query labels
stay local; the manifest records source revisions, hashes and selected IDs.
See [the evaluation protocol](benchmarks/ACADEMIC_EVAL.md) for metrics and limits.
