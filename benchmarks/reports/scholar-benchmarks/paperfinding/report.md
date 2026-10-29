# Sampled PaperFindingBench semantic evaluation

12 sampled validation questions; independent GPT-4o rubric evaluation; five results / 1500 estimated tokens.

| Method | P@5 (grade >=2) | Pooled nDCG@5 | Aspect coverage | Useful non-Exa papers/query |
|---|---:|---:|---:|---:|
| single:exa | 0.4833 | 0.7287 | 0.9028 | 0.000 |
| single:openalex | 0.3333 | 0.5886 | 0.8417 | 1.000 |
| single:semantic_scholar | 0.2333 | 0.3195 | 0.4250 | 0.833 |
| single:parallel | 0.1333 | 0.2845 | 0.5667 | 0.333 |
| rrf | 0.5333 | 0.7533 | 0.9028 | 1.000 |
| jev | 0.6333 | 0.9291 | 0.9500 | 1.167 |
| decider | 0.6000 | 0.8972 | 0.9056 | 1.000 |
| voyage | 0.6167 | 0.9075 | 0.9389 | 0.917 |
| exa+jev | 0.5333 | 0.7825 | 0.9028 | 0.000 |
| exa+voyage | 0.4833 | 0.7783 | 0.9028 | 0.000 |

## Scope and limitations

- This is a bounded-output adaptation, not an official AstaBench leaderboard result. Model judgments remain provisional without human adjudication.
- Official per-question criteria, pinned evaluator model, weighted grading thresholds, and normalizer are used. Criterion names normalize case and underscores to avoid the reported scorer bug. Missing/invalid judgments stop evaluation.
- Every unique final-result paper is judged once on the same 400-token source representation, irrespective of method. Provider/ranker identity and scores are hidden. Official known-positive overrides receive grade3. The rankers are not their own judges.
- Pooled nDCG uses the union of judged final results as its ideal set; it is not exhaustive corpus recall. P@5 treats grades2 and3 as useful, and missing output slots as zero. Weighted aspect coverage is diagnostic, not proof that distinct papers add distinct findings.
- Useful non-Exa papers means grade>=2 papers absent from all raw retrieved Exa results, including date-excluded hits, not merely absent from its final five. Exact normalized-title matches conservatively exclude possible same-paper versions from novelty credit.
- All model blends rank the same provider-balanced shortlist. Exa+Jev and Exa+Voyage isolate reranking benefits from adding providers. Baselines share merged identities and text metadata.
- Results lacking verifiable pre-June-2025 dates are excluded. This can disproportionately remove provider results with poor metadata; see date-audit.jsonl and per-provider accepted counts.
- Semantic Scholar search failed with HTTP429 on 3/12 questions; metadata enrichment failed with HTTP429 on 4/12. A separately disclosed post-hoc metadata-success sensitivity analysis is in diagnostics.json; it does not replace this full-sample result.
- Independent evaluator sees source passages/abstracts, not synthesized answers. Passage usefulness of the final shortened excerpt requires a separate assessment.
- Source snapshots, full rankings, evaluator responses, query bootstrap intervals, output tokens, and separate spend ledgers are retained. Official adjusted-score adaptation is in JSON; its estimated recall and rank correction have different semantics from ordinary nDCG.
