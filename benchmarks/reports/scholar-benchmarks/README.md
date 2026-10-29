# Sampled scholar evaluations

These experiments were sampled before scoring. No settings were selected on their outcomes, and no default ranker was changed. COVID was not evaluated.

## PaperFindingBench: does adding providers help?

Twelve semantic validation questions sampled with seed 20261029 from the pinned official release. No exact question overlap was found with prior saved experiments. Exa, OpenAlex, Semantic Scholar, and Parallel each receive the same original question (with adapter-specific deterministic retrieval handling). All global model rankers receive one identical provider-balanced shortlist of at most 20 papers and a 400-estimated-token document budget. Final evidence contains up to five papers under 1500 estimated tokens.

An independent pinned GPT-4o evaluator applies the released per-question criteria to each unique pooled final-result paper, hiding provider/model identities and scores. The three evaluated rankers never judge their own outputs. This is a bounded-output adaptation, not an official AstaBench leaderboard submission or exhaustive corpus evaluation.

| Method | Useful-paper P@5 | Pooled nDCG@5 | Weighted aspect coverage |
|---|---:|---:|---:|
| Exa | 0.4833 | 0.7287 | 0.9028 |
| Exa + Jev | 0.5333 | 0.7825 | 0.9028 |
| Exa + Voyage | 0.4833 | 0.7783 | 0.9028 |
| RRF blend | 0.5333 | 0.7533 | 0.9028 |
| Blend + Jev | 0.6333 | 0.9291 | 0.9500 |
| Blend + Decider | 0.6000 | 0.8972 | 0.9056 |
| Blend + Voyage | 0.6167 | 0.9075 | 0.9389 |

Useful means grade >=2 on the upstream 0–3 document scale. Missing result slots count against P@5. Pooled nDCG normalizes against the assessed union, not every relevant paper in the world.

The Voyage blend improves useful-paper P@5 over Exa+Voyage by 0.1333 (paired-query bootstrap 95% interval [0.0667, 0.2000]). The Jev blend improves over Exa+Jev by 0.1000, with interval approximately [0, 0.2000]. Model-to-model differences remain uncertain: Jev minus Voyage is 0.0167, interval [-0.1167, 0.1500]. Intervals are unadjusted exploratory comparisons on a small sample.

The Jev blend retains 14 useful question-paper pairs absent from all raw Exa results across the twelve questions, conservatively excluding exact-title version matches. This is additional relevant-paper coverage, not merely more provider names. For example, VideoGPT (https://arxiv.org/abs/2104.10157) is contributed by OpenAlex and Parallel for the autoregressive transformer video-generation query. Its original abstract explicitly describes autoregressive modeling of video latents with a GPT-like architecture.

There are material limitations. Semantic Scholar search failed with HTTP429 on three questions, and metadata enrichment failed on four. The historical June2025 cutoff removes undated results, which affects providers differently. Of raw results, date verification accepts 67/119 Exa, 102/120 OpenAlex, 53/70 Semantic Scholar, and 17/120 Parallel records. These counts must accompany the result; this is not evidence of universal superiority on current unrestricted search.

A post-hoc sensitivity analysis keeps the eight questions with successful metadata enrichment, using unchanged rankings and labels: Exa P@5 0.55, Exa+Jev 0.625, Exa+Voyage 0.55, Jev blend 0.65, Voyage blend 0.65. Voyage blend versus Exa+Voyage remains +0.10, bootstrap interval [0.025, 0.175]; Jev blend versus Exa+Jev is only +0.025, interval [-0.075, 0.125]. This subset is a diagnostic, not a fresh validation.

The runner shares the core provider, fusion, ranker, and presentation code. It uses the provider-balanced shortlist and global ranking, without experimental final score-band diversity reordering or Exa anchoring. These results do not validate every public configuration or change the library defaults.

See [full results and caveats](paperfinding/report.md), [query-level results and intervals](paperfinding/results.json), [operational sensitivity analysis](paperfinding/diagnostics.json), and [frozen sampling manifest](paperfinding/manifest.json).

## DORIS-MAE: which ranker handles the same papers best?

Sixteen questions sampled from the upstream 60-query track; all 1774 native candidates retained. Identical question and bounded title/abstract representations are used across every method. These are fixed candidate pools, so this experiment cannot measure an Exa-versus-blend retrieval advantage.

| Method | Linear nDCG@5 | Linear nDCG@10 | Aspect coverage@5 |
|---|---:|---:|---:|
| BM25 | 0.5806 | 0.5946 | 0.7506 |
| Random | 0.4986 | 0.5084 | 0.7185 |
| Jev | 0.7324 | 0.7503 | 0.7668 |
| Decider | 0.7537 | 0.7608 | 0.7752 |
| Voyage | 0.7668 | 0.7817 | 0.8023 |

All three models improve over BM25 on this sample. Voyage has the highest mean, but Voyage minus Jev has bootstrap interval [-0.0031, 0.0690]. This does not establish a universal best ranker. Labels are released AnnoGPT aspect judgments rather than independent human gold, and 58.3% of abstracts exceed the shared 400-estimated-token budget. The dataset was verified against the publisher's MD5 checksum.

See [full ranking report](doris/report.md) and [paired comparisons](doris/results.json). Abstracts and original judgments are retained in the frozen snapshot; the gigabyte source dataset and response caches are excluded from packaging and Git.

## Cost and reproduction

New reported costs plus the Parallel estimate total $0.87923; the session total including prior experiments is approximately $1.89069, below the $5 ceiling. This is not an invoice; interrupted in-flight native requests can have small unreported costs. All attempted-request reservations and reported costs are retained in separate ledgers. See [cost accounting](budget.json).

Free report replay:

```sh
python -m benchmarks.scholar_eval report --out benchmarks/reports/scholar-benchmarks/doris
python -m benchmarks.paperfinding_eval report
```

Native loading and paid reproduction commands are documented in [benchmark instructions](../../README.md). Dataset access and benchmark dependencies are only needed for explicitly requested evaluations; ordinary search does not download benchmark data. No COVID evaluation or further paid calls are required for these reports.
