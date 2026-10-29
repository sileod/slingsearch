import csv
from . import Query, rows
from slingsearch.models import Result, Source
from slingsearch.fusion import canonical_url


def read_csv(path):
    with open(path, newline='') as f:
        return list(csv.DictReader(f))


def load(results_path, judgments_path, queries_path, *, repetition=1, variant='primary', runs_path=None):
    questions = {r['query_id']: r for r in rows(queries_path)}
    queries = {}
    for r in read_csv(results_path):
        if int(r['repetition']) != repetition or r['query_variant'] != variant:
            continue
        qid = r['query_id']
        question = questions[qid]['query' if variant == 'primary' else 'reference_query']
        if not question:
            raise ValueError('Missing query text for selected variant')
        q = queries.setdefault(qid, Query(qid, question, provenance='Web Search Benchmark Kit: released pooled judgments'))
        url = canonical_url(r['canonical_url'] or r['url'])
        q.candidates.append(Result(url, r['title'], url, passages=[r['snippet']],
                                   sources=[Source(r['provider'], int(r['rank']))]))
    for r in read_csv(judgments_path):
        if r['query_id'] in queries:
            q = queries[r['query_id']]
            url = canonical_url(r['canonical_url'])
            grade = int(r['relevance_grade'])
            if url in q.qrels and q.qrels[url] != grade:
                raise ValueError('Select a single assessor or adjudicated judgment file')
            q.qrels[url] = grade
    if runs_path:
        for r in read_csv(runs_path):
            if int(r['repetition']) == repetition and r['query_variant'] == variant and r['query_id'] in queries:
                queries[r['query_id']].retrieval[r['provider']] = {
                    'latency': float(r['latency_ms']) / 1000, 'success': r['http_status'] == '200', 'cost': None}
    return list(queries.values())
