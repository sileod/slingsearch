# Scholar retrieval and Exa-only reranking ablations

Eight exposed development questions and eight new questions (seed 17, excluding all earlier scholar queries). Native OpenAlex semantic search, 10 results per provider. No known relevant papers were inserted.

Provider identities are resolved once across the full pool and shared by every baseline. This prevents richer DOI metadata from creating a false ensemble recall gain. Cross-provider identity resolution includes compatible title/year matches. Gold relevance is then matched by DOI/identifier.

| Split | Exa nDCG@5 | OpenAlex semantic | Exa-preserving blend | RRF | Exa Recall@10 | Blend Recall@10 |
|---|---:|---:|---:|---:|---:|---:|
| development | 0.5000 | 0.4077 | 0.6414 | 0.6414 | 0.5000 | 0.7500 |
| holdout | 0.4917 | 0.3277 | 0.4784 | 0.4729 | 0.6250 | 0.6250 |

Semantic OpenAlex adds known-target papers for two development questions (SemEval-2022 Task 7 and BERT), but adds no known-target papers missing from Exa on the new eight-question sample. Consequently the retrieval gain did not replicate on that sample.

The predeclared Exa-preserving rule keeps Exa top 1 and uses linear rank support with weight 1 for the tail. It was not selected on new-query labels.

## Exa-only reranking on the eight new questions

One global ranker receives Exa’s own title/snippets only, with the unchanged original relevance rubric and 300 estimated tokens per candidate. Shared document identities are used only for evaluation. These three backend comparisons and keep-first variants are exploratory; no winner was chosen before seeing the results.

| Method | nDCG@5 | nDCG@10 | Recall@10 |
|---|---:|---:|---:|
| exa | 0.4917 | 0.5126 | 0.6250 |
| exa+decider | 0.4557 | 0.4752 | 0.6250 |
| exa+decider+keep_first | 0.4727 | 0.4923 | 0.6250 |
| exa+jev | 0.5312 | 0.5312 | 0.6250 |
| exa+jev+keep_first | 0.5143 | 0.5143 | 0.6250 |
| exa+voyage | 0.5018 | 0.5227 | 0.6250 |
| exa+voyage+keep_first | 0.4727 | 0.4936 | 0.6250 |

Exa + Jev improves nDCG@5 by 0.0395 in this sample, with paired 95% bootstrap interval [-0.0199, 0.1384]. This is promising but inconclusive; the interval includes no improvement and multiple variants were examined. No claim that Jev is generally best is supported.

## Fix and limits

The previous OpenAlex adapter used `search=` with full natural-language questions. Native keyword search treats unseparated terms as AND; the adapter now defaults to `search.semantic=`. An explicit `semantic=False` adapter option retains the keyword-search baseline. Semantic search supports at most 50 results and requires pacing at 1 request/second. The collection used sequential paced requests.

Sources: [OpenAlex semantic search](https://help.openalex.org/api/semantic-search/) and [keyword-search semantics](https://help.openalex.org/api/searching/).

LitSearch relevance labels are incomplete for broad searches; query samples are small and drawn from its first 100 published questions. Metadata-based document linking can be imperfect. The eight new questions were not used to tune fusion parameters, but subsequent model comparisons are exploratory. No product ranking defaults were changed.

Retrieval ablation reported cost: $0.07200000. Exa-only reranking reported cost: $0.002683594. Total known reported experiment spend including preceding runs: $0.157750076, below the $5 session ceiling.

Frozen inputs, gold identifier metadata, result orders, per-query metrics, costs and confidence intervals are in the adjacent JSON files. No credentials are stored.
