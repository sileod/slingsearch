# Sampled method comparison

20 frozen web queries with heuristic labels; 8 live LitSearch questions. Seed 42. Scholar matching uses exact DOI/identifiers only; no relevant papers injected. Semantic Scholar search failed on all 8 queries and is retained as a failed baseline. Ranking-only latencies/costs are not end-to-end costs.

Offline replay: model rows measure ranking latency/cost only; single rows use released retrieval telemetry.
Unknown costs/timings are shown as NA. Unjudged documents count as nonrelevant. Best single is selected by aggregate nDCG@5 on this evaluation set (optimistic).

web

| Method | nDCG@5 | Recall@10 | Latency p50 (s) | Cost/1000 queries | Success |
|---|---:|---:|---:|---:|---:|
| Best single (exa) | 0.6976 | 0.2277 | 1.1228 | NA | 1.0000 |
| rrf | 0.7420 | 0.2567 | 0.0008 | 0.0000 | 1.0000 |
| jev | 0.7119 | 0.2683 | 0.9263 | 0.4434 | 1.0000 |
| decider | 0.7153 | 0.2631 | 1.1315 | 0.1184 | 1.0000 |
| voyage | 0.7280 | 0.2603 | 0.2589 | 0.0576 | 1.0000 |

scholar

| Method | nDCG@5 | Recall@10 | Latency p50 (s) | Cost/1000 queries | Success |
|---|---:|---:|---:|---:|---:|
| Best single (exa) | 0.5000 | 0.5000 | 0.8303 | 7.0000 | 1.0000 |
| rrf | 0.3616 | 0.5000 | 0.0005 | 0.0000 | 1.0000 |
| jev | 0.3772 | 0.5000 | 0.7826 | 0.4218 | 1.0000 |
| decider | 0.3914 | 0.5000 | 1.1683 | 0.1178 | 1.0000 |
| voyage | 0.3366 | 0.5000 | 0.2775 | 0.0650 | 1.0000 |

Total known reported spend including earlier validation: $0.083066482.

Evaluation correction: DOI URLs normalized before gold matching. Every provider ordering and model candidate ID/text pool was verified unchanged; saved rankings were rescored without new paid calls.

## Blending and reranking conclusions

Primary metric: nDCG@5, higher is better. The eight-provider web replay includes
Brave, Context, Exa, Kagi session, Octen, Parallel, Perplexity and Tavily. The
separate default-trio check uses Exa, Parallel and Brave from the same snapshots.
Model rows all rerank the same top-20 RRF shortlist with the same bounded text.

| Method | Web (20 questions) | Scholar (8 questions) |
|---|---:|---:|
| Exa alone (best sampled individual) | 0.6976 | 0.5000 |
| RRF default web trio | 0.7314 | — |
| RRF all available providers | 0.7420 | 0.3616 |
| RRF + Jev | 0.7119 | 0.3772 |
| RRF + Decider | 0.7153 | 0.3914 |
| RRF + Voyage | 0.7280 | 0.3366 |

Blending improved web nDCG@5 by 0.0445 over Exa for the eight-provider blend,
but its paired 95% bootstrap interval [-0.0312, 0.1353] includes no improvement.
The default three-provider blend is also promising but inconclusive. The scholar
blend reduced quality relative to Exa alone; none of its rerankers recovered the
Exa baseline. OpenAlex contributed no exact-ID known-relevant hits and Semantic
Scholar search failed for every sampled query. This sample does not establish
that either provider is generally weak: natural-question retrieval, API failures,
and incomplete relevance labels limit the conclusion.

Jev is not established as the best reranker. Voyage had the best web nDCG@5
among the model backends; Decider had the best scholar nDCG@5. Jev had the best
web nDCG@10 (0.7436), so the ranking also depends on the chosen cutoff. No model
beat free RRF on web nDCG@5. Jev remains the requested default, not a benchmark
winner. No defaults were changed based on this sample.

Web qrels are heuristic labels, not independent human relevance assessments.
Scholar relevance uses exact DOI/ID matching only; no title guesses, no injected
positives. Missing relevant identifiers and unjudged but useful papers can cause
false negatives. Eight scholar queries and twenty web queries remain small.
The best individual provider was selected on this same sample, which is an
optimistic baseline. Bootstrap intervals do not account for that selection.
Single-provider results have ten documents while the fused shortlist has twenty;
Recall@20 therefore compares different retrieval depths. Rerankers share the
same candidate set, so improvements at rank 20 cannot increase their recall.

## Scholar failure audit

Historical Semantic Scholar status contained four HTTP errors and four parser
`KeyError` failures. Live reinspection confirmed the API can return HTTP 200
with `{"total": 0, "offset": 0}` and no `data` field. The adapter now treats that
as successful empty retrieval, with a regression test. A separate live check
returned HTTP 429. The historical artifacts are preserved; the four parser
errors must not be interpreted as upstream outages. None supplied candidates.

Every deduplicated scholar pool contained 10–20 results, so the top-20 shortlist
excluded no documents. Exa ranked each of the four exactly matched known papers
first. For the wikiHowToImprove question (litsearch-28), only Exa supplied
results: Jev moved the target from #1 to #5, Decider to #3 and Voyage to #2.
This demotion therefore cannot be explained by adding OpenAlex results or by
RRF shortlist truncation. Model score differences, supplied evidence and the
mismatch between question relevance and known-item labels need further study.
