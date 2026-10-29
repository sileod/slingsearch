# Provider contribution audit

Scholar defaults retain Exa, Parallel, OpenAlex, Semantic Scholar and Google Scholar via SerpAPI. The last two receive half-weight fusion votes and one candidate reservation each, with no final-result quota. This is an exploratory policy, not a validated optimum.

| Provider | Known-positive query-paper pairs | Unique pairs | Search success |
|---|---:|---:|---:|
| exa | 10 | 6 | 34/34 |
| parallel | 2 | 1 | 34/34 |
| openalex | 9 | 7 | 34/34 |
| semantic_scholar | 0 | 0 | 24/34 |
| serpapi | 1 | 0 | 34/34 |
| papers_with_code | 0 | 0 | 34/34 |
| huggingface | 0 | 0 | 34/34 |

| RRF provider set | SAGE known-paper Hit@10 | SAGE open WeightedRecall@10 | LitQA Recall@30 |
|---|---:|---:|---:|
| exa | 0.167 | 0.069 | 0.500 |
| exa_openalex | 0.083 | 0.135 | 0.600 |
| previous_default | 0.083 | 0.135 | 0.600 |
| pruned_default | 0.167 | 0.124 | 0.600 |
| weighted_scholar | 0.167 | 0.124 | 0.600 |
| all_sources | 0.083 | 0.060 | 0.600 |

- Exploratory policy selection on 34 reused queries, not independent validation
- SAGE positives are incomplete; zero observed label contributions does not establish general uselessness
- Shared identity metadata is held constant; query wording and provider result depths are not equal-cost retrieval budgets
- GitHub implementation discovery is separate from paper retrieval and is not assessed by these relevance labels

Reproduce without API calls: `python -m benchmarks.provider_audit`. Private academic snapshots are required. Only aggregate counts are published, not query-paper ground truth.
