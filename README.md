# SlingSearch

Multi-provider web and academic search for agents. Ask a question; get ranked,
source-backed excerpts with URLs, identifiers and provider provenance. No generated answer.

Search providers run concurrently. Results are deduplicated, fused with RRF,
then ranked once against the original question. **Jev Latest is the default
ranker**, including for novelty and related-work searches. Voyage, D1, Decider,
ensembles and free RRF are explicit alternatives.

## Install and search

```sh
pip install 'git+https://github.com/sileod/slingsearch.git'
# Set OPENROUTER_API_KEY and keys for your selected retrieval providers.
slingsearch "What limits diffusion models for time-series forecasting?" --mode scholar
```

```python
from slingsearch import search

web = search("What evidence supports incremental builds reducing CI latency?")
print(web.to_markdown())

papers = search(
    "What limits diffusion models for time-series forecasting?",
    mode="scholar",
    min_relevance=4,
    max_results=5,
    token_budget=1500,
)
print(papers.to_markdown())
```

`asearch()` has the same options. Use `Client` for pooled connections and explicit
`Config`. JSON includes provider status, actual ranking backend, model, timings,
usage, reported costs and warnings.

## Only keep useful sources

`min_relevance=4` / `--min-relevance 4` filters the combined shortlist using:

| Score | Meaning |
|---|---|
| 1 | Unrelated |
| 2 | Topically related; unlikely to help |
| 3 | Partial supporting evidence |
| 4 | Directly addresses a substantial part of the question |
| 5 | Directly answers the question with substantial evidence |

The default cutoff is **3/5**: sources need at least partial supporting evidence.
Use **4/5** for stricter searches. Searches can return fewer than `max_results`,
including zero. Set `min_relevance=None` or `--no-relevance-filter` to opt out.
Scores are model judgments, **not calibrated probabilities**. Inspect
`usage.relevance_filter` for scores and qualification status. The judgment uses
bounded shortlist excerpts; output budgeting may shorten them further.

Jev, D1 and Decider reuse their rubric scores without another model call.
Voyage, ensembles and RRF require an additional packed Jev assessment to use
this cutoff; native scores are not interchangeable. This can increase cost and
latency. If scoring fails, the default returns no qualified results and an explicit
warning. To retrieve unscored leads deliberately, disable the cutoff; ranking
failures then return RRF with a warning. For fully free search, use
`ranker="rrf", min_relevance=None` (CLI: `--ranker rrf --no-relevance-filter`).

## Optional source requirements

```python
papers = search(
    "How do retrieval systems generalize across domains?", mode="scholar",
    checks=[
        "Does the paper evaluate on held-out datasets?",
        "Does it report results against a named baseline?",
    ],
)
```

All checks must pass (`check_threshold=.7` by default). These separate packed
Jev judgments use the source excerpts; missing evidence does not count as yes.
They add model calls even when relevance scores are already available. Estimates
are not calibrated probabilities or full-paper verification. Results preserve
ranking order and expose per-check scores in `usage.checks`. Failed scoring
returns no qualified sources. CLI: repeat `--check "..."`; works in the review
APIs and MCP too. No checks are imposed by default.

## Providers and defaults

- **Web:** Exa, Parallel, Brave.
- **Scholar:** Exa, Parallel, OpenAlex, Semantic Scholar, Google Scholar via SerpAPI.
- **Opt-in:** Tavily, Firecrawl, GitHub, Papers with Code, Hugging Face.

Missing credentials disable the corresponding provider with a warning. Set
`EXA_API_KEY`, `PARALLEL_API_KEY`, `BRAVE_API_KEY`, `OPENALEX_API_KEY`,
`SEMANTIC_SCHOLAR_API_KEY`, `SERPAPI_API_KEY`, `TAVILY_API_KEY`, or
`FIRECRAWL_API_KEY`. GitHub and Hugging Face accept optional `GITHUB_TOKEN` and
`HF_TOKEN`. No global configuration file is required.

Override with `providers=["exa", "openalex"]`. Scholar fusion gives Semantic
Scholar and SerpAPI half-weight votes and one shortlist opportunity each;
there are no guaranteed final provider slots. This policy is exploratory.
`Config(scholar_provider_weights={})` restores plain RRF without reservations.
Citation counts never boost relevance. Packing is opt-in through
`Config(decision_batch_size=5)`; the default remains one pending broader validation.

## Existing related work and novelty checks

```python
from slingsearch import related_work_search, novelty_search, load_papers

known = load_papers("related_work.md")  # stable paper links, or structured JSON
report = related_work_search(known, contribution="Your precise research claim")
print(report.to_markdown())

report = novelty_search(
    "Your precise research claim", related_work=known,
    claims=["First contribution", "Second contribution"],
)
```

These bounded workflows resolve known papers, search for missing work, optionally
expand citation graphs, and judge claim overlap on displayed evidence. Known
papers remain distinguished from discoveries. Missing results do not establish
novelty. `min_relevance` also works here; overlap and relevance are separate judgments.

## Evaluation

Results are mixed; the blend is **not established as better than the best
individual provider**. A frozen exploratory sample of 24 SAGE and 10 LitQA queries:

| Retrieval/fusion method | SAGE known-paper Hit@10 | SAGE open weighted recall@10 | LitQA Recall@30 |
|---|---:|---:|---:|
| Exa | .167 | .069 | .500 |
| Weighted scholar blend | .167 | .124 | .600 |

OpenAlex alone scored .145 on the open-ended sample, above the blend.
These are fusion results, not evidence that Jev or the cutoff improves quality.
The half weights were explored on reused queries, not independently validated.
A separate 12-query DORIS pilot scored Voyage .768 versus packed Jev .758 nDCG@10;
it is too small to establish a winner. Jev is an operational default, not a proven best reranker.

```sh
pip install 'slingsearch[bench] @ git+https://github.com/sileod/slingsearch.git'
python -m benchmarks.provider_audit  # free replay; requires saved private snapshots
```

See the [provider audit](benchmarks/reports/provider-audit/README.md),
[academic evaluation](benchmarks/reports/academic-samples/README.md), and
[full reference](docs/reference.md) for datasets, reproduction, limitations,
advanced options and benchmark commands. Benchmark datasets are never downloaded
by ordinary searches. The default cutoff is a conservative product policy, not an empirically calibrated optimum; no new quality claim is made for it yet.

## Agent integration

Use the Python API, JSON CLI (`--format json`), or optional stdio MCP server:

```sh
pip install 'slingsearch[mcp] @ git+https://github.com/sileod/slingsearch.git'
slingsearch-mcp
```

An installable [agent skill](skills/slingsearch/SKILL.md) covers mode selection,
cutoffs, citation discipline and partial failures. Copy `skills/slingsearch/` into
your agent's skill directory; Codex discovers it under `~/.codex/skills/`.

MIT license. Search and model API usage may incur charges.
