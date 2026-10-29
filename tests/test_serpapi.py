import httpx
import pytest
from slingsearch import Client, Config
from slingsearch.providers.serpapi import SerpAPI


@pytest.mark.parametrize('mode', ['web', 'scholar'])
async def test_serpapi_contract_and_pagination(mode):
    calls = []
    def handle(request):
        calls.append(request)
        assert request.url.params['engine'] == ('google_scholar' if mode == 'scholar' else 'google')
        assert request.url.params['q'] == ('question' if mode == 'scholar' else 'research question')
        assert request.url.params['api_key'] == 'fixture'
        if mode == 'scholar': assert request.url.params['as_ylo'] == '2020'
        index = len(calls)
        return httpx.Response(200, json={'organic_results': [{
            'title': f'Paper {index}', 'link': f'https://example.org/{index}', 'snippet': 'Original evidence',
            'publication_info': {'summary': 'A Author - Journal, 2023', 'authors': [{'name': 'A Author'}]},
            'inline_links': {'cited_by': {'total': 42, 'cites_id': f'cluster{index}'}},
            'resources': [{'link': f'https://example.org/{index}.pdf'}]}],
            'serpapi_pagination': {'next': 'https://untrusted.invalid/?api_key=secret'} if index == 1 else {}})
    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        results = await SerpAPI(client, 'fixture', mode, (2020, 2025)).search('research question', 2)
    assert len(calls) == 2 and calls[1].url.params['start'] == '2'
    assert results[1].sources[0].rank == 2
    assert results[0].authors == ['A Author'] and results[0].year == 2023
    assert results[0].citation_count == 42 and results[0].passages == ['Original evidence']
    assert results[0].related_urls == ['https://example.org/1.pdf']


async def test_serpapi_error_does_not_leak_query_key():
    def handle(request):
        return httpx.Response(200, json={'error': 'private fixture-key'})
    async with Client(Config(api_keys={'serpapi': 'fixture-key'}), transport=httpx.MockTransport(handle)) as client:
        response = await client.asearch('question', providers=['serpapi'], ranker='rrf', min_relevance=None)
    assert response.failed == {'serpapi': 'ValueError'}
    assert 'fixture-key' not in response.to_json()


async def test_serpapi_skips_missing_urls():
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(200, json={
        'organic_results': [{'title': 'Citation only'}, {'title': 'PDF only', 'resources': [{'link': 'https://example.org/p.pdf'}]}]}))) as client:
        results = await SerpAPI(client, 'fixture', 'scholar').search('question')
    assert len(results) == 1 and results[0].url.endswith('.pdf')
