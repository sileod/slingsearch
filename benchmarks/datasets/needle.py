"""Load NEEDLE engine reports with document judgments, not answer-hit proxies."""
import json
from . import Query
from slingsearch.models import Result, Source
from slingsearch.fusion import canonical_url


def load(path, mode='web'):
    report = json.loads(open(path).read())
    queries = {}
    for provider, engine in report['engines'].items():
        for row in engine['per_query']:
            question = row['query']
            q = queries.setdefault(question, Query(question, question, mode, provenance='NEEDLE: provider-maintained judgments'))
            q.retrieval[provider] = {'success': not bool(row.get('search_error')),
                                     'latency': row.get('latency_s'), 'cost': row.get('cost_usd')}
            for rank, item in enumerate(row.get('results', []), 1):
                url = canonical_url(item['url'])
                q.candidates.append(Result(url, item.get('title', ''), url,
                                           passages=[item.get('snippet') or ''], sources=[Source(provider, rank)]))
                grade = item.get('rating')
                if grade is not None:
                    grade = int(grade)
                    if url in q.qrels and q.qrels[url] != grade:
                        raise ValueError('Conflicting judgments for one query/URL; adjudicate before replay')
                    q.qrels[url] = grade
    if not any(q.qrels for q in queries.values()):
        raise ValueError('This report has no document relevance labels; answer-hit ranks are not document qrels')
    return list(queries.values())
