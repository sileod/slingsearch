import csv
from dataclasses import asdict
import json
import httpx
import pytest
from benchmarks.datasets import Query, load_snapshot
from benchmarks.datasets import needle, websearch_kit, litsearch, sparbench
from benchmarks.evaluate import bootstrap_delta, match_paper, metrics
from benchmarks.compare import compare, paired, report
from benchmarks.live import snapshot
from slingsearch import Config, Result, Source


def write(tmp_path, name, data):
    path = tmp_path / name
    path.write_text(json.dumps(data))
    return path


def test_needle(tmp_path):
    path = write(tmp_path, 'needle.json', {'engines': {'exa': {'per_query': [
        {'query': 'q', 'results': [{'url': 'https://x.org', 'title': 'X', 'snippet': 'text', 'rating': 4}]}]}}})
    result = needle.load(path)[0]
    assert result.qrels == {'https://x.org/': 4}
    assert result.candidates[0].sources == [Source('exa', 1)]
    path = write(tmp_path, 'no-labels.json', {'engines': {'exa': {'per_query': [
        {'query': 'q', 'hit_rank': 1, 'results': [{'url': 'https://x.org'}]}]}}})
    with pytest.raises(ValueError, match='no document'):
        needle.load(path)


def test_kit(tmp_path):
    def csvfile(name, rows):
        path = tmp_path / name
        with path.open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        return path
    results = csvfile('results.csv', [{'query_id': 'q', 'query_variant': 'primary', 'repetition': 1,
        'canonical_url': 'https://x.org', 'url': 'https://x.org', 'title': 'X', 'snippet': 'text', 'provider': 'exa', 'rank': 1}])
    judgments = csvfile('judgments.csv', [{'query_id': 'q', 'canonical_url': 'https://x.org', 'relevance_grade': 3}])
    queries = write(tmp_path, 'queries.json', [{'query_id': 'q', 'query': 'question'}])
    q = websearch_kit.load(results, judgments, queries)[0]
    assert q.question == 'question' and q.qrels['https://x.org/'] == 3


def test_litsearch(tmp_path):
    q = litsearch.load(write(tmp_path, 'lit.json', [{'query': 'question', 'corpusids': [123, 456]}]))[0]
    assert q.mode == 'scholar' and q.qrels == {'corpusid:123': 1, 'corpusid:456': 1}
    assert not q.candidates


def test_sparbench(tmp_path):
    path = write(tmp_path, 'spar.json', [{'query': 'question', 'answers': [
        {'paperID': '2406.04286', 'title': 'Paper', 'abstract': 'Evidence', 'human_grade': 3, 'Qwen25_72B_score': .8}]}])
    q = sparbench.load(path)[0]
    assert not q.qrels and q.candidates[0].identifiers == {'arxiv': '2406.04286'}
    assert sparbench.load(path, grade_field='human_grade')[0].qrels == {'https://arxiv.org/abs/2406.04286': 3}
    with pytest.raises(ValueError, match='integer'):
        sparbench.load(path, grade_field='Qwen25_72B_score')


def test_matching_and_metrics():
    result = Result('x', 'The paper', 'https://x.org', identifiers={'corpusid': '123'})
    scores = metrics([result], {'corpusid:123': 1, 'other': 0})
    assert scores['nDCG@5'] == 1 and scores['R@10'] == 1
    assert metrics([], {'corpusid:123': 1})['R@10'] == 0
    assert match_paper(result, {'corpusid:123': {'identifiers': {'corpusid': '123'}}}) == 'corpusid:123'
    assert match_paper(result, {'a': {'title': 'The paper'}}) is None
    delta = bootstrap_delta({'a': 1, 'b': 1}, {'a': 0, 'b': 0})
    assert delta['ci95'] == [1, 1]


async def test_comparison_same_pool_and_no_silent_fallback(tmp_path):
    q = Query('q', 'question', candidates=[
        Result('a', 'A', 'https://a.org', ['Evidence A'], sources=[Source('exa', 1)]),
        Result('b', 'B', 'https://b.org', ['Evidence B'], sources=[Source('parallel', 1)])],
        qrels={'https://a.org/': 3, 'https://b.org/': 1})
    path = tmp_path / 'snapshot.jsonl'
    path.write_text(json.dumps(asdict(q)) + '\n')
    loaded = load_snapshot(path)
    records = await compare(loaded, config=Config(api_keys={}))
    assert len({r['pool_hash'] for r in records}) == 1
    assert all(not r['success'] for r in records if r['method'] in ['jev', 'decider', 'voyage'])
    assert '| rrf |' in report(records) and '| jev | NA' in report(records)
    assert paired(records) == {}


async def test_live_snapshot_no_keys_saved():
    def handle(request):
        return httpx.Response(200, json={'results': [{'title': 'A', 'url': 'https://a.org', 'text': 'Evidence'}]})
    q = Query('q', 'question')
    await snapshot([q], ['exa', 'brave'], config=Config(api_keys={'exa': 'private-secret'}), transport=httpx.MockTransport(handle))
    assert len(q.candidates) == 1 and q.retrieval['brave']['skipped']
    assert 'private-secret' not in json.dumps(asdict(q))


async def test_failed_single_provider_is_not_dropped():
    q = Query('q', 'question', candidates=[Result('a', 'A', 'https://a.org', sources=[Source('exa', 1)])],
              qrels={'https://a.org/': 1}, retrieval={'brave': {'success': False}})
    records = await compare([q], methods=['rrf'], config=Config(api_keys={}))
    failed = next(r for r in records if r['method'] == 'single:brave')
    assert not failed['success'] and failed['metrics']['nDCG@5'] == 0


def test_balanced_sample_excludes_conflicting_labels(tmp_path):
    from benchmarks.sample import web_sample
    def table(name, data):
        path = tmp_path / name
        with path.open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(data[0])); w.writeheader(); w.writerows(data)
        return path
    results = table('results.csv', [dict(query_id=q, query_variant='primary', repetition=1,
        canonical_url=f'https://example.org/{q}', url=f'https://example.org/{q}',
        title=q, snippet='evidence', provider='exa', rank=1) for q in ['a','b','c']])
    judgments = table('judgments.csv', [dict(query_id=q, canonical_url=f'https://example.org/{q}', relevance_grade=g)
        for q,g in [('a',3),('b',2),('c',3),('c',0)]])
    catalog = write(tmp_path, 'queries.json', [dict(query_id=q, query=q, category=cat)
        for q,cat in [('a','known_item'),('b','research'),('c','research')]])
    sample, info = web_sample(results, judgments, catalog, size=2)
    assert {q.id for q in sample} == {'a','b'}
    assert info['conflicting_qrel_queries_excluded'] == 1
    again,_ = web_sample(results, judgments, catalog, size=2)
    assert [q.id for q in sample] == [q.id for q in again]


async def test_live_doi_url_matches_gold_before_snapshot():
    def handle(request):
        return httpx.Response(200, json={'results': [{'title': 'Paper', 'url': 'https://doi.org/10.1234/PAPER', 'text': 'Evidence'}]})
    q = Query('q', 'question', mode='scholar', qrels={'corpusid:42': 1})
    await snapshot([q], ['exa'], config=Config(api_keys={'exa':'fixture'}),
                   corpus={'corpusid:42': {'doi':'10.1234/paper', 'identifiers':{}}},
                   transport=httpx.MockTransport(handle))
    assert q.candidates[0].identifiers['corpusid'] == '42'
    assert metrics(q.candidates, q.qrels)['R@10'] == 1


async def test_single_baselines_share_cross_provider_document_identity():
    q = Query('q', 'question', mode='scholar', qrels={'corpusid:42': 1}, candidates=[
        Result('a', 'The paper', 'https://example.org/preprint', year=2025,
               identifiers={'arxiv':'2501.00001'}, sources=[Source('exa', 1)]),
        Result('b', 'The paper', 'https://doi.org/10.1234/paper', doi='10.1234/paper',
               year=2025, identifiers={'corpusid':'42'}, sources=[Source('openalex', 1)])])
    records = await compare([q], methods=['rrf'], config=Config(api_keys={}))
    assert all(r['metrics']['R@10'] == 1 for r in records)


def test_anchor_preserves_exa_first_and_selection_ignores_holdout_labels():
    from copy import deepcopy
    from benchmarks.anchor import anchor
    from benchmarks.anchor_compare import experiment
    candidates = [Result('a','A','https://a.org',sources=[Source('exa',1)]),
                  Result('b','B','https://b.org',sources=[Source('exa',2),Source('parallel',1),Source('brave',1)])]
    assert anchor(candidates,keep=1,weight=2)[0].title == 'A'
    dev = Query('dev','q',candidates=candidates,qrels={'https://a.org/':1,'https://b.org/':3})
    holdout = Query('holdout','q',candidates=deepcopy(candidates),qrels={'https://a.org/':3,'https://b.org/':1})
    first=experiment([dev,holdout],{'dev'})['selection']
    holdout.qrels={'https://a.org/':0,'https://b.org/':3}
    second=experiment([dev,holdout],{'dev'})['selection']
    assert first == second
