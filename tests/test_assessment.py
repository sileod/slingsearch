import json
import httpx
import pytest
from slingsearch import Client, Config


async def test_packed_assessment_after_voyage_preserves_order_and_scores():
    calls = []
    def handle(request):
        body = json.loads(request.content) if request.content else {}
        if request.url.host == 'api.exa.ai':
            return httpx.Response(200, json={'results': [
                {'title': 'Paper A', 'url': 'https://example.org/a', 'text': 'Original evidence A'},
                {'title': 'Paper B', 'url': 'https://example.org/b', 'text': 'Original evidence B'}]})
        if request.url.path.endswith('/rerank'):
            return httpx.Response(200, json={'results': [{'index': 0, 'relevance_score': .9}, {'index': 1, 'relevance_score': .8}],
                                            'usage': {'cost': .001}})
        calls.append(body)
        assert body['model'] == '~typesafe/jev-latest' and body['state'] == {'question': 'question'}
        answers = {}
        for name, item in body['questions'].items():
            assert 'Original evidence' in item['instructions']['document']
            answers[name] = {'score': 3} if item['type'] == 'score' else {'noul': .9 if name.endswith('has_answer_evidence') else .1}
        return httpx.Response(200, json={'model': 'typesafe/jev-1.14', 'answers': answers, 'usage': {'cost': .002}})
    async with Client(Config(api_keys={'exa': 'fixture', 'openrouter': 'fixture'}), transport=httpx.MockTransport(handle)) as client:
        result = await client.asearch('question', providers=['exa'], ranker='voyage', assess=True, min_relevance=None)
    assert len(calls) == 1 and len(calls[0]['questions']) == 6
    assert [r.title for r in result.results] == ['Paper A', 'Paper B']
    assert [r.model_score for r in result.results] == [.9, .8]
    assert result.evidence_assessment['useful_result_count'] == 2
    assert result.evidence_assessment['has_potential_evidence'] is True
    assert result.cost_usd == pytest.approx(.003)


async def test_assessment_missing_key_is_unknown_and_empty_search_is_explicit():
    def handle(request):
        return httpx.Response(200, json={'results': [{'title': 'A', 'url': 'https://example.org/a', 'text': 'Original evidence'}]})
    async with Client(Config(api_keys={'exa': 'fixture'}), transport=httpx.MockTransport(handle)) as client:
        result = await client.asearch('question', providers=['exa'], ranker='rrf', assess=True, min_relevance=None)
        empty = await client.asearch('question', providers=[], ranker='rrf', assess=True, min_relevance=None)
    assert result.evidence_assessment['status'] == 'unavailable'
    assert 'has_potential_evidence' not in result.evidence_assessment
    assert empty.evidence_assessment['status'] == 'no_results'
