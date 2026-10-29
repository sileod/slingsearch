# Scholar diversity: retrieval contribution and final retention

Known-paper relevance, actual output capped at five results / 1,500 estimated tokens. No final provider quotas. Shared identities and global judgments are used across comparisons.

Diversity policy selected on development queries: `global_balanced`. Held-out labels did not select it.

development

| Method | Queries | nDCG@5 | Recall@5 | New relevant papers retained / available | Mean output tokens |
|---|---:|---:|---:|---:|---:|
| diverse_0.05 | 36 | 0.4515 | 0.5880 | 3 / 3 | 1500 |
| diverse_0.1 | 36 | 0.4479 | 0.5880 | 3 / 3 | 1500 |
| diverse_0.2 | 36 | 0.4195 | 0.5046 | 3 / 3 | 1500 |
| exa | 36 | 0.5430 | 0.5741 | 0 / 3 | 1496 |
| global_balanced | 36 | 0.4612 | 0.5880 | 3 / 3 | 1499 |
| global_exa_view | 36 | 0.4550 | 0.5139 | 0 / 3 | 1500 |
| global_rrf | 36 | 0.4612 | 0.5880 | 3 / 3 | 1499 |
| rrf | 36 | 0.5208 | 0.6296 | 2 / 3 | 1500 |

heldout

| Method | Queries | nDCG@5 | Recall@5 | New relevant papers retained / available | Mean output tokens |
|---|---:|---:|---:|---:|---:|
| diverse_0.05 | 20 | 0.2558 | 0.4500 | 2 / 3 | 1500 |
| diverse_0.1 | 20 | 0.2414 | 0.4500 | 2 / 3 | 1500 |
| diverse_0.2 | 20 | 0.2427 | 0.4500 | 2 / 3 | 1500 |
| exa | 20 | 0.3504 | 0.4000 | 0 / 3 | 1500 |
| global_balanced | 20 | 0.2746 | 0.4500 | 2 / 3 | 1500 |
| global_exa_view | 20 | 0.3150 | 0.3750 | 0 / 3 | 1500 |
| global_rrf | 20 | 0.2746 | 0.4500 | 2 / 3 | 1500 |
| rrf | 20 | 0.3549 | 0.4250 | 1 / 3 | 1500 |

Novel papers not found by Exa:

- litsearch-81 (development): [SemEval-2022 Task 7: Identifying Plausible Clarifications of Implicit and Underspecified Phrases in Instructional Texts](https://doi.org/10.18653/v1/2022.semeval-1.146), found by openalex, parallel.
- litsearch-94 (development): [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://doi.org/10.18653/v1/n19-1423), found by openalex, parallel.
- litsearch-7 (heldout): [M](https://arxiv.org/pdf/2203.10282), found by parallel.
- litsearch-51 (heldout): [MiniLMv2: Multi-Head Self-Attention Relation Distillation for Compressing Pretrained Transformers](https://doi.org/10.18653/v1/2021.findings-acl.188), found by openalex.
- litsearch-52 (development): [Don’t Parse, Insert: Multilingual Semantic Parsing with Insertion Based Decoding O X](https://arxiv.org/pdf/2010.03714), found by parallel.
- litsearch-97 (heldout): [Open Language Learning for Information Extraction](http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.259.3048), found by openalex.

Known-paper qrels are incomplete; only annotated non-Exa papers count as useful diversity. Shared identities avoid false recall gains. Actual output obeys 1500 estimated tokens, with at most five results.

Model scores were computed once over the complete unique pool (up to 40 papers), then reused across candidate-selection policies. Independent Decisions judgments make this reuse exact at the request level. Main pool comparisons use the same top-20 count and 300-token document representation. Provider failures remain in the collection snapshot.
