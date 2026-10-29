import json
import httpx
import pytest
from slingsearch import Config, Result
from slingsearch.rankers import make_ranker
from slingsearch.rankers.ensemble import fuse_orders


def test_rank_fusion_ignores_incompatible_raw_scores():
    docs = [Result(x, x, 'https://example.org/' + x, rrf_score=i / 100) for i, x in enumerate('ABC')]
    docs[0].model_score = 999
    ranked = fuse_orders([docs, docs[::-1]], docs)
    assert [r.id for r in ranked] == ['C', 'A', 'B']
    assert docs[0].model_score == 999 and docs[1].model_score is None
    with pytest.raises(ValueError): fuse_orders([[docs[0], docs[0], docs[2]]], docs)


@pytest.mark.parametrize('members', [('jev',), ('jev', 'jev'), ('jev', 'ensemble'), ('unknown', 'voyage')])
def test_invalid_ensemble_members(members):
    with pytest.raises(ValueError, match='ensemble_rankers'): Config(ensemble_rankers=members)


@pytest.mark.parametrize('failure', [False, True])
async def test_d1_and_ensemble_contracts_and_settled_failures(failure):
    calls = []
    def handle(request):
        body = json.loads(request.content)
        calls.append(body)
        if 'documents' in body:
            return httpx.Response(200, json={'model': body['model'], 'results': [
                {'index': i, 'relevance_score': 100 - i} for i in range(2)],
                'usage': {'input_tokens': 20, 'cost': .002}})
        assert body['model'] == '~liquid/d1-ultra-latest'
        if failure:
            return httpx.Response(500)
        return httpx.Response(200, json={'model': 'liquid/d1-ultra',
            'answers': {k: {'score': i} for i, k in enumerate(body['questions'])},
            'usage': {'input_tokens': 30, 'cost': .001}})
    config = Config(api_keys={'openrouter': 'fixture'}, ensemble_rankers=('d1', 'voyage'), decision_batch_size=2)
    docs = [Result(x, x, 'https://example.org/' + x) for x in 'AB']
    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        ranker = make_ranker('ensemble', client, config)
        if failure:
            with pytest.raises(ValueError, match='ensemble rankers failed'): await ranker.rank('question', docs)
            assert ranker.cost_usd == .002
        else:
            ranked = await ranker.rank('question', docs)
            assert {r.id for r in ranked} == {'A', 'B'}
            assert ranker.cost_usd == pytest.approx(.003)
            assert ranker.usage['input_tokens'] == 50
            assert 'liquid/d1-ultra' in ranker.resolved_models
        assert len(calls) == 2
        assert all(r.model_score is None for r in docs)


@pytest.mark.parametrize('failure', [False, True])
async def test_ensemble_public_pipeline_provenance_and_fallback(failure):
    from slingsearch import Client
    def handle(request):
        body = json.loads(request.content)
        if request.url.host == 'api.exa.ai':
            return httpx.Response(200, json={'results': [{'title': 'Evidence', 'url': 'https://example.org', 'text': 'Original evidence'}]})
        if 'documents' in body:
            return httpx.Response(200, json={'results': [{'index': 0, 'relevance_score': .9}], 'usage': {'cost': .001}})
        if failure:
            return httpx.Response(503)
        return httpx.Response(200, json={'model': 'liquid/d1-ultra', 'answers': {'relevance': {'score': 3}}})
    config = Config(api_keys={'exa': 'fixture', 'openrouter': 'fixture'}, ensemble_rankers=('d1', 'voyage'))
    async with Client(config, transport=httpx.MockTransport(handle)) as client:
        response = await client.asearch('question', providers=['exa'], ranker='ensemble', min_relevance=None)
    assert response.requested_ranker == 'ensemble'
    assert '~liquid/d1-ultra-latest' in response.requested_model
    assert response.usage['ensemble_members']['voyage']['success']
    assert response.ranking_backend == ('rrf' if failure else 'ensemble')
    assert response.results[0].model_score is None if failure else response.results[0].model_score > 0
    if failure:
        assert any('returned RRF' in w for w in response.warnings)
