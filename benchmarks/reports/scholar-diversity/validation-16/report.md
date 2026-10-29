# Fresh scholar-diversity validation

The Exa-preserving policy was fixed after the exploratory 56-query run, then tested on 16 previously unused questions. This is a negative validation result. Do not interpret the experimental feature as a proven quality improvement.

| Method | nDCG@5 | Recall@5 | Relevant non-Exa papers retained / available |
|---|---:|---:|---:|
| exa | 0.4807 | 0.5563 | 0 / 3 |
| rrf | 0.5954 | 0.6188 | 0 / 3 |
| global_balanced | 0.2616 | 0.4813 | 1 / 3 |
| guarded_diverse | 0.4414 | 0.5312 | 1 / 3 |

The guarded diversity policy finds one annotated relevant paper absent from Exa, but loses other known-paper coverage and ranking quality. It therefore does not establish a successful blend. The paired nDCG@5 difference versus Exa is -0.0393, with 95% bootstrap interval [-0.1331, 0.0484].

Relevance-only global Jev ranking is worse on this known-paper test. Plain RRF improves the sampled ranking of existing hits, but retains no additional non-Exa known targets in its final five. Neither provider count nor a higher score on one metric establishes useful literature diversity.

## What the policy does

1. Retrieve 10 results each from Exa, semantic OpenAlex, lexical Semantic Scholar, and Parallel with a scholarly objective. Record failures.
2. Deduplicate once and share document identity mappings across all baselines.
3. Reserve provider coverage in a 20-document candidate pool.
4. Apply one global Jev scoring stage with the full question and bounded title/excerpts.
5. Prefer nonredundant evidence within narrow score bands; preserve Exa’s best available shortlisted result through an explicit retrieval prior.
6. Return at most five sources under the same 1,500 estimated-token payload budget.

## Useful diversity audit

- litsearch-26: [Stance Detection in COVID-19 Tweets](https://doi.org/10.18653/v1/2021.acl-long.127), found by openalex; retained by none of the final-five methods.
- litsearch-26: [P-Stance: A Large Dataset for Stance Detection in Political Domain](https://doi.org/10.18653/v1/2021.findings-acl.208), found by openalex; retained by none of the final-five methods.
- litsearch-45: [[2010.06028] End-to-End Synthetic Data Generation for Domain Adaptation of Question Answering Systems](https://arxiv.org/abs/2010.06028), found by parallel; retained by global_balanced, guarded_diverse.

Limitations: LitSearch labels identify known relevant papers and are incomplete for broader literature discovery. This evaluation cannot establish that every unjudged alternative is useless. However, that limitation is not evidence that the blend is good. A separate manual review of additional papers and passages is still required. Native provider failures remain part of the operational result.

Known reported spend across the session: $0.651467. Parallel estimate for the 72 new basic-search requests: $0.360, using [official listed pricing](https://docs.parallel.ai/getting-started/pricing). Combined estimate: $1.011467, below $5.

The feature remains opt-in. No ranking or provider defaults were changed on the basis of this experiment.
