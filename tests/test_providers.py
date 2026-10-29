import json
import httpx
import pytest
import respx
from slingsearch.providers import PROVIDERS
from slingsearch.models import Source

CASES = [
 ('exa', 'POST', 'https://api.exa.ai/search', {'results': [{'title': 'Paper', 'url': 'https://example.org/p', 'highlights': ['evidence']}]}),
 ('parallel', 'POST', 'https://api.parallel.ai/v1/search', {'results': [{'title': 'Paper', 'url': 'https://example.org/p', 'excerpts': ['evidence']}]}),
 ('brave', 'GET', 'https://api.search.brave.com/res/v1/web/search', {'web': {'results': [{'title': 'Paper', 'url': 'https://example.org/p', 'description': 'evidence'}]}}),
 ('tavily', 'POST', 'https://api.tavily.com/search', {'results': [{'title': 'Paper', 'url': 'https://example.org/p', 'content': 'evidence'}]}),
 ('firecrawl', 'POST', 'https://api.firecrawl.dev/v2/search', {'success': True, 'data': {'web': [{'title': 'Paper', 'url': 'https://example.org/p', 'description': 'evidence'}]}}),
 ('openalex', 'GET', 'https://api.openalex.org/works', {'results': [{'id': 'https://openalex.org/W1', 'doi': 'https://doi.org/10.1/test', 'title': 'Paper', 'publication_year': 2025, 'abstract_inverted_index': {'evidence': [0]}, 'authorships': [{'author': {'display_name': 'Author'}}], 'cited_by_count': 8, 'primary_location': {'source': {'display_name': 'Venue'}}}]}),
 ('semantic_scholar', 'GET', 'https://api.semanticscholar.org/graph/v1/paper/search', {'data': [{'paperId': 'p1', 'title': 'Paper', 'url': 'https://example.org/p', 'abstract': 'evidence', 'externalIds': {'DOI': '10.1/test', 'CorpusId': 123}, 'authors': [{'name': 'Author'}], 'year': 2025, 'citationCount': 8}]}),
]


@pytest.mark.parametrize('name,method,url,payload', CASES)
@respx.mock
async def test_contract(name, method, url, payload):
    route = respx.route(method=method, url=url).mock(return_value=httpx.Response(200, json=payload))
    async with httpx.AsyncClient() as client:
        result = (await PROVIDERS[name](client, 'test-secret', 'scholar').search('question', 1))[0]
    assert result.title == 'Paper'
    assert result.sources == [Source(name, 1)]
    assert 'evidence' in (result.abstract or ' '.join(result.passages))
    request = route.calls[0].request
    if method == 'POST':
        body = json.loads(request.content)
        if name == 'parallel':
            assert body['objective'].endswith('question') and body['search_queries'] == ['question']
        else:
            assert body.get('query', body.get('objective')) == 'question'
        assert 'test-secret' not in request.content.decode()
    else:
        parameter = {'brave': 'q', 'semantic_scholar': 'query', 'openalex': 'search.semantic'}[name]
        assert request.url.params[parameter] == 'question'
    if name in ('openalex', 'semantic_scholar'):
        assert result.year == 2025 and result.authors == ['Author']


@respx.mock
async def test_scholar_pagination():
    route = respx.get('https://api.semanticscholar.org/graph/v1/paper/search')
    route.side_effect = [httpx.Response(200, json={'data': [{'paperId': '1', 'title': 'one'}], 'next': 1}),
                         httpx.Response(200, json={'data': [{'paperId': '2', 'title': 'two'}]})]
    async with httpx.AsyncClient() as client:
        result = await PROVIDERS['semantic_scholar'](client).search('q', 2)
    assert len(result) == 2 and result[1].sources[0].rank == 2
    assert route.calls[1].request.url.params['offset'] == '1'


@respx.mock
async def test_semantic_scholar_valid_empty_response_without_data():
    route = respx.get('https://api.semanticscholar.org/graph/v1/paper/search').mock(
        return_value=httpx.Response(200, json={'total': 0, 'offset': 0}))
    async with httpx.AsyncClient() as client:
        result = await PROVIDERS['semantic_scholar'](client).search('a long literature question', 10)
    assert result == [] and route.call_count == 1


@respx.mock
async def test_openalex_semantic_default_and_explicit_keyword_baseline():
    route = respx.get('https://api.openalex.org/works').mock(return_value=httpx.Response(200, json={'results': []}))
    async with httpx.AsyncClient() as client:
        await PROVIDERS['openalex'](client).search('Which papers explain causal discovery?', 10)
        await PROVIDERS['openalex'](client, semantic=False).search('causal discovery', 10)
        with pytest.raises(ValueError, match='at most 50'):
            await PROVIDERS['openalex'](client).search('question', 51)
    assert route.calls[0].request.url.params['search.semantic'] == 'Which papers explain causal discovery?'
    assert 'search' not in route.calls[0].request.url.params
    assert route.calls[1].request.url.params['search'] == 'causal discovery'


@respx.mock
async def test_semantic_scholar_uses_lexical_query_and_preserves_question_input():
    route = respx.get('https://api.semanticscholar.org/graph/v1/paper/search').mock(
        return_value=httpx.Response(200, json={'total':0,'offset':0}))
    question = 'Can you recommend papers using few-shot NLI for relation extraction?'
    async with httpx.AsyncClient() as client:
        await PROVIDERS['semantic_scholar'](client).search(question)
    query = route.calls[0].request.url.params['query']
    assert query == 'few shot NLI relation extraction'
    assert question.startswith('Can you recommend')


@respx.mock
async def test_scholarly_rate_limit_retries_once_with_bounded_delay(monkeypatch):
    waits = []
    async def sleep(seconds):
        waits.append(seconds)
    monkeypatch.setattr('slingsearch.providers.base.asyncio.sleep', sleep)
    route = respx.get('https://api.semanticscholar.org/graph/v1/paper/search')
    route.side_effect = [httpx.Response(429, headers={'Retry-After':'2'}),
                         httpx.Response(200, json={'total':0,'offset':0})]
    async with httpx.AsyncClient() as client:
        assert await PROVIDERS['semantic_scholar'](client).search('question') == []
    assert waits == [2] and route.call_count == 2


@respx.mock
async def test_long_rate_limit_does_not_stall_search():
    route = respx.get('https://api.openalex.org/works').mock(return_value=httpx.Response(429, headers={'Retry-After':'120'}))
    async with httpx.AsyncClient() as client:
        with pytest.raises(httpx.HTTPStatusError):
            await PROVIDERS['openalex'](client).search('question')
    assert route.call_count == 1
