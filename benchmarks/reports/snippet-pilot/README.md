# Semantic Scholar snippet pilot

Six-query live pilot: 6/6 paper searches, 6/6 snippet searches succeeded.

| Method | Unique papers, summed per query | Body snippets |
|---|---:|---:|
| papers | 4 | 0 |
| snippets | 51 | 50 |

| Sample / metric | Paired queries | Paper search | Snippet search |
|---|---:|---:|---:|
| sage_short / Hit@10 | 2 | 0.0 | 0.5 |
| sage_open / WeightedRecall@10 | 2 | 0.0 | 0.0 |
| litqa2 / Recall@30 | 2 | 0.0 | 0.5 |

Enrichment: {"successful": 6, "requested_papers": 26, "enriched_papers": 12, "selected_body_passages": 11, "changed_papers": 9}

- Six reused questions; not independent validation or an established improvement
- Live provider results can change; raw normalized snapshots retained privately
- Paper search uses existing ten-term lexical cleanup; snippets use the full question with an explicit literal-asterisk HTTP-400 retry. This changes query strategy as well as index
- Equal limit=10 counts papers in paper search but snippets in snippet search; fewer unique snippet papers are possible
- LitQA historical filtering conservatively excludes 2024 and later papers; no publication-date lookup in this pilot
- No model-based relevance or passage usefulness assessment; key lifetime limit blocks fresh model calls
- Enrichment uses a fixed five-paper frozen shortlist and one restricted request; papers may receive no snippet
- Body-text availability and changed excerpts are coverage measurements, not evidence-quality judgments

Reproduce: `python -m benchmarks.snippet_eval` with private academic snapshots and a Semantic Scholar key.
API contract: https://api.semanticscholar.org/api-docs/snippets
