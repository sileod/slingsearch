# Priority academic retrieval evaluations

## SAGE (highest priority)

Official release: https://github.com/HughieHu/Sage
Paper: https://arxiv.org/abs/2602.05975

The release contains 1,200 questions: 600 short-form and 600 open-ended,
150 of each in computer science, healthcare, humanities and natural science.
Start with three questions per stratum (24 total), fixed seed and source revision.
Keep the remaining questions untouched while developing methods.

Short-form evaluates recovery of the known paper: report Hit@5/10/20/30 and
MRR@10. Open-ended has two grades: seed papers (2) and shared references (1).
Report the paper's weighted reference recall, calculated as recovered grade sum
at K divided by total reference grade sum; also report nDCG@5/10 and Recall@10/20
as supplementary metrics. Do not substitute exponential nDCG gains for the
linear weighted-recall metric. Report each task and domain separately.

These labels are constructed from seed/reference relationships. Other useful
papers may be unlabelled; they are not independently assessed negatives. This
benchmark alone cannot prove the usefulness of newly discovered sources or
academic diversity. The paper describes generated queries, despite the repository
calling them expert-curated. SAGE's corpus is closed; live API results must be
labelled an open-web adaptation, not compared directly with corpus leaderboard
scores. Source versions disagree on corpus size, so pin versions rather than
assuming the abstract's rounded 200,000 documents.

## AstaBench LitQA2-FullText-Search (highest academic priority)

Official task: https://github.com/allenai/asta-bench/tree/main/astabench/evals/paper_finder
Overview: https://allenai.org/asta/bench

Use the official full-text subset and mapping of LAB-bench question IDs to
Semantic Scholar CorpusIds. The official validation split has 10 questions,
the test split 75. Start with all validation questions and leave test untouched.
Primary metric: Recall@30; report Recall@5/10/20 and MRR@10 separately.
Score paper retrieval only, never answer correctness. No LLM relevance judge is
needed for the known target match. Correct answers, distractors, key passages and
source titles must never enter the retrieval or ranking input.

The official corpus insertion cutoff is 2024-10-17. Publication date filtering is
not equivalent to corpus insertion membership. A live API adaptation cannot claim
strict official reproduction without verifying original index membership. A
paper match also does not demonstrate that our selected passage contains the
answer: assess passage usefulness separately.

## Shared comparison protocol

Save normalized retrieval from Exa, Parallel, OpenAlex and Semantic Scholar once.
Compare each individual provider, retrieval RRF, and global Jev/D1/Voyage ranking
on identical frozen shortlists and text budgets. Include the strongest individual
provider **with the same ranker**; raw Exa alone is an insufficient baseline.
Evaluate experimental ranker ensembles on those same candidate sets, using
predeclared equal-weight rank fusion. Record attempted failures rather than
substituting RRF under the requested model's name.

Report retrieval-pool recall before ranking, shortlist recall after truncation,
and final ranking recall. Use a 50-document shortlist when testing Recall@30;
a 20-document shortlist cannot establish 30-document retrieval performance.
Keep the ordinary five-result/1,500-token product view separate from an expanded
30-document evaluation output. Use paired bootstrap intervals and per-domain
counts. Run new API calls only under the remaining session-wide budget; preparing
these datasets performs no search or paid model calls.

Match exact DOI/Semantic Scholar/CorpusId aliases first. Resolve missing metadata
without exposing gold titles to retrievers. Title-only matches require manual
verification or unique title plus compatible year; ambiguous cases stay unmatched
and are reported. Keep source revisions and hashes in the manifest; downloaded
questions, labels and source text stay outside committed reports.

## Preparation and status

```sh
python -m benchmarks.academic_samples --output benchmarks/reports/academic-samples
```

This downloads query files and LAB-bench questions, not the multi-gigabyte SAGE
corpus or embedding indices. It parses official Asta mapping literals without
executing downloaded code. The loaders have synthetic contract tests and are
also checked against the downloaded official releases. The resulting sample is
an evaluation input, **not a completed evaluation or evidence of a model win**.
