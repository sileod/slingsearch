import asyncio
import json
import httpx
import pytest
from slingsearch import Client, Config, Result, Source
from slingsearch.fusion import canonical_url, fuse
from slingsearch.output import budget_results, evidence_tokens, markdown, tokens
from slingsearch.passages import select
from slingsearch.rankers import make_ranker


def candidates():
    return fuse([Result('a', 'Alpha', 'https://example.org/a', ['Direct evidence'], sources=[Source('exa', 1)]),
                 Result('b', 'Beta', 'https://example.org/b', ['Other evidence'], sources=[Source('parallel', 1)])])


def test_dedup_and_provenance():
    rows = [Result('1', 'Paper', 'https://x.org/a?utm_source=x#top', doi='https://doi.org/10.1/X', sources=[Source('exa', 2)]),
            Result('2', 'Paper', 'https://y.org/b', doi='10.1/x', sources=[Source('parallel', 5)]),
            Result('3', 'Other', 'https://doi.org/10.1/x', sources=[Source('exa', 4)])]
    result = fuse(rows)
    assert len(result) == 1
    assert len(result[0].sources) == 3
    assert result[0].rrf_score == pytest.approx(1 / 62 + 1 / 65)
    assert rows[0].doi == 'https://doi.org/10.1/X'  # no caller mutation
    assert canonical_url('https://EXAMPLE.org:443/a?b=2&utm_source=x&a=1#foo') == 'https://example.org/a?a=1&b=2'
    assert canonical_url('https://example.org/A/') != canonical_url('https://example.org/a')


def test_title_dedup_and_year():
    rows = [Result(str(i), 'A paper!', f'https://x.org/{i}', abstract='abstract', year=year)
            for i, year in enumerate([2024, 2024, 2025])]
    assert len(fuse(rows)) == 2
    assert len(fuse([Result('1', 'Home', 'https://a.org'), Result('2', 'Home', 'https://b.org')])) == 2


def test_passages_are_extracts_and_budgeted():
    text = ('Diffusion forecasting has limitations in calibration and long horizons. ' * 200)
    passages = select('forecasting limitations', [text])
    assert all(p in text and tokens(p) <= 180 for p in passages)
    result = candidates()[0]
    result.passages = passages
    for budget in (1, 50, 200, 1500):
        selected = budget_results([result], 5, budget)
        assert evidence_tokens(selected) <= budget
        assert all(p in text for r in selected for p in r.passages)


@pytest.mark.parametrize('backend', ['jev', 'decider', 'voyage'])
async def test_rank_contract_and_identical_text(backend):
    requests = []
    def handle(request):
        body = json.loads(request.content)
        requests.append(body)
        if request.url.path.endswith('decisions'):
            assert body['questions']['relevance']['type'] == 'score'
            assert len(body['questions']['relevance']['criteria']) == 5
            score = 4 if body['state']['document'].startswith('Beta') else 1
            return httpx.Response(200, json={'answers': {'relevance': {'score': score}}, 'model': body['model'] + '-resolved', 'usage': {'input_tokens': 50, 'cost': .001}})
        return httpx.Response(200, json={'results': [{'index': 1, 'relevance_score': .9}, {'index': 0, 'relevance_score': .2}], 'usage': {'total_tokens': 100}})
    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        ranker = make_ranker(backend, client, Config(api_keys={'openrouter': 'secret'}))
        result = await ranker.rank('question', candidates())
    assert result[0].title == 'Beta'
    assert candidates()[0].model_score is None
    assert ranker.usage


async def test_pipeline_partial_failure_and_fallback():
    def handle(request):
        if request.url.host == 'api.exa.ai':
            return httpx.Response(200, json={'results': [{'title': 'Evidence', 'url': 'https://x.org', 'text': 'source words'}]})
        raise httpx.ConnectError('secret-leak', request=request)
    config = Config(api_keys={'exa': 'secret', 'parallel': 'secret', 'openrouter': 'secret'})
    async with Client(config, transport=httpx.MockTransport(handle)) as client:
        result = await client.asearch('question', providers=['exa', 'parallel', 'brave'])
    assert result.successful == ['exa']
    assert result.failed == {'parallel': 'ConnectError'}
    assert 'brave' in result.skipped
    assert result.ranking_backend == 'rrf' and result.model is None
    assert result.results[0].model_score is None
    assert 'secret' not in result.to_json()
    assert any('returned RRF' in w for w in result.warnings)


async def test_missing_ranking_key():
    def handle(request):
        return httpx.Response(200, json={'results': [{'id': 'https://openalex.org/W1', 'title': 'Paper', 'abstract_inverted_index': {'Useful': [0], 'evidence': [1]}}]})
    async with Client(Config(api_keys={}), transport=httpx.MockTransport(handle)) as client:
        result = await client.asearch('question', mode='scholar', providers=['openalex'])
    assert result.ranking_backend == 'rrf'
    assert result.results[0].passages == ['Useful evidence']


async def test_invalid_rerank_mapping_falls_back():
    def handle(request):
        if request.url.host == 'api.exa.ai':
            return httpx.Response(200, json={'results': [{'title': 'A', 'url': 'https://a.org'}, {'title': 'B', 'url': 'https://b.org'}]})
        return httpx.Response(200, json={'results': [{'index': 0, 'relevance_score': 1}, {'index': 0, 'relevance_score': 2}]})
    async with Client(Config(api_keys={'exa': 'x', 'openrouter': 'x'}), transport=httpx.MockTransport(handle)) as client:
        result = await client.asearch('q', providers=['exa'], ranker='voyage')
    assert result.ranking_backend == 'rrf'


def test_sync_reuse():
    def handle(request):
        return httpx.Response(200, json={'results': []})
    with Client(Config(api_keys={'exa': 'x'}), transport=httpx.MockTransport(handle)) as client:
        for _ in range(2):
            assert client.search('q', providers=['exa'], ranker='rrf').successful == ['exa']


async def test_sync_in_loop():
    from slingsearch import search
    with pytest.raises(RuntimeError, match='await asearch'):
        search('q')

async def test_all_rankers_receive_identical_candidate_text():
    texts_by_model = {}
    def handle(request):
        body = json.loads(request.content)
        if request.url.path.endswith('decisions'):
            texts_by_model.setdefault(body['model'], []).append(body['state']['document'])
            return httpx.Response(200, json={'answers': {'relevance': {'score': 3}}})
        texts_by_model[body['model']] = body['documents']
        return httpx.Response(200, json={'results': [{'index': i, 'relevance_score': .5} for i in range(len(body['documents']))]})
    config = Config(api_keys={'openrouter': 'fixture-only'}, candidate_tokens=20)
    pool = candidates()
    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        for name in ('jev', 'decider', 'voyage'):
            await make_ranker(name, client, config).rank('same question', pool)
    representations = list(texts_by_model.values())
    assert representations[0] == representations[1] == representations[2]
    assert all(tokens(t) <= 20 for t in representations[0])


async def test_decision_failure_settles_requests_and_preserves_rrf():
    completed = []
    async def handle(request):
        body = json.loads(request.content)
        text = body['state']['document']
        if text.startswith('Alpha'):
            return httpx.Response(503)
        await asyncio.sleep(.01)
        completed.append(text)
        return httpx.Response(200, json={'answers': {'relevance': {'score': 4}}, 'usage': {'cost': .001}})
    pool = candidates()
    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        ranker = make_ranker('jev', client, Config(api_keys={'openrouter': 'fixture-only'}))
        with pytest.raises(ValueError, match='judgments failed'):
            await ranker.rank('q', pool)
    assert len(completed) == 1 and all(r.model_score is None for r in pool)
    assert ranker.cost_usd == .001


def test_output_budget_shares_evidence_across_papers():
    rows = [Result(str(i),f'Paper {i}',f'https://example.org/{i}',
                   passages=['Distinct source evidence. ' * 100], sources=[Source('exa' if i<3 else 'openalex',i+1)])
            for i in range(5)]
    selected = budget_results(rows,5,1500)
    assert len(selected) == 5 and all(r.passages for r in selected)
    assert evidence_tokens(selected) <= 1500
    assert all(r.passages[0] in rows[i].passages[0] for i,r in enumerate(selected))


async def test_diversity_reserves_candidate_slots_before_one_ranking_stage():
    def handle(request):
        if request.url.host == 'api.exa.ai':
            return httpx.Response(200,json={'results':[{'title':name,'url':f'https://example.org/{name}','text':'source evidence'} for name in ['a','b','c']]})
        return httpx.Response(200,json={'results':[{'id':'https://openalex.org/W1','title':'different paper','abstract_inverted_index':{'different':[0],'evidence':[1]}}]})
    config=Config(api_keys={'exa':'fixture','openalex':'fixture'})
    async with Client(config,transport=httpx.MockTransport(handle)) as client:
        response=await client.asearch('question',mode='scholar',providers=['exa','openalex'],ranker='rrf',shortlist=2,diversity=True)
    assert response.usage['diversity_enabled'] is True
    assert response.usage['candidate_providers'] == ['exa','openalex']
    assert response.usage['candidate_count'] == 2
    assert len(response.results) == 2 and response.usage['output_tokens'] <= 1500


async def test_invalid_diversity_option_fails_before_paid_retrieval():
    def handle(request):
        raise AssertionError('No retrieval should occur')
    async with Client(Config(api_keys={'exa':'fixture'}),transport=httpx.MockTransport(handle)) as client:
        with pytest.raises(ValueError,match='diversity must be a boolean'):
            await client.asearch('question',providers=['exa'],diversity='yes')


async def test_anchor_must_be_a_selected_provider_before_retrieval():
    def handle(request):
        raise AssertionError('No network should be called')
    async with Client(Config(api_keys={'exa':'fixture'}),transport=httpx.MockTransport(handle)) as client:
        with pytest.raises(ValueError,match='anchor_provider'):
            await client.asearch('question',providers=['exa'],anchor_provider='openalex')
