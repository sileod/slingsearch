"""Deterministic samples from local native benchmark releases; no downloads."""
import csv
from collections import defaultdict
import json
from pathlib import Path
import random
from .datasets import Query, rows
from .datasets.websearch_kit import read_csv
from slingsearch.fusion import canonical_url
from slingsearch.models import Result, Source


def web_sample(results_path, judgments_path, queries_path, runs_path=None, *, size=20, seed=42):
    catalog = {r['query_id']: r for r in rows(queries_path)}
    labels, conflicts = defaultdict(dict), set()
    for row in read_csv(judgments_path):
        qid, url, grade = row['query_id'], canonical_url(row['canonical_url']), int(row['relevance_grade'])
        if url in labels[qid] and labels[qid][url] != grade:
            conflicts.add(qid)
        labels[qid][url] = grade
    groups = defaultdict(list)
    for row in read_csv(results_path):
        if row['query_variant'] == 'primary' and int(row['repetition']) == 1:
            groups[row['query_id']].append(row)
    eligible = [qid for qid in sorted(groups) if qid not in conflicts and labels[qid] and any(labels[qid].values())]
    # Round-robin shuffled categories to avoid a sample dominated by one intent.
    categories = defaultdict(list)
    for qid in eligible:
        categories[catalog[qid]['category']].append(qid)
    rng = random.Random(seed)
    for category in sorted(categories):
        rng.shuffle(categories[category])
    selected = []
    while len(selected) < min(size, len(eligible)):
        for category in sorted(categories):
            if categories[category] and len(selected) < size:
                selected.append(categories[category].pop())
    queries = []
    for qid in selected:
        q = Query(qid, catalog[qid]['query'], qrels=labels[qid],
                  provenance='Web Search Benchmark Kit rep-1-20260805; auto-heuristic labels; seed-42 category-balanced sample; conflicting-qrel queries excluded')
        for row in groups[qid]:
            url = canonical_url(row['canonical_url'] or row['url'])
            q.candidates.append(Result(url, row['title'], url, passages=[row['snippet']],
                                       sources=[Source(row['provider'], int(row['rank']))]))
        queries.append(q)
    if runs_path:
        index = {q.id: q for q in queries}
        for r in read_csv(runs_path):
            if r['query_id'] in index and r['query_variant'] == 'primary' and int(r['repetition']) == 1:
                index[r['query_id']].retrieval[r['provider']] = {'latency': float(r['latency_ms']) / 1000,
                                                              'success': r['http_status'] == '200', 'cost': None}
    return queries, {'eligible_queries': len(eligible), 'conflicting_qrel_queries_excluded': len(conflicts & set(groups)),
                      'selected_ids': selected, 'seed': seed}
