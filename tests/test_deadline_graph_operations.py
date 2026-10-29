import asyncio
import json
import httpx
import pytest
from slingsearch import Client, Config


async def test_fast_cancels_straggler_and_keeps_provenance():
    cancelled = []
    async def handle(request):
        if request.url.host == 'api.search.brave.com':
            try: await asyncio.sleep(10)
            except asyncio.CancelledError:
                cancelled.append('brave'); raise
        if request.url.host == 'api.exa.ai':
            return httpx.Response(200, json={'results': [{'title': 'Paper A', 'url': 'https://example.org/a', 'text': 'source evidence'}]})
        return httpx.Response(200, json={'results': [{'title': 'Paper B', 'url': 'https://example.org/b', 'excerpts': ['other evidence']}]})
    async with Client(Config(api_keys={p: 'fixture' for p in ['exa', 'parallel', 'brave']}), transport=httpx.MockTransport(handle)) as client:
        result = await client.asearch('question', providers=['exa', 'parallel', 'brave'], ranker='rrf',
                                      fast=True, deadline=1, fast_min_results=2, fast_grace=0, min_relevance=None)
    assert cancelled == ['brave']
    assert result.cancelled == {'brave': 'fast_results_ready'}
    assert set(result.successful) == {'exa', 'parallel'} and len(result.results) == 2
    assert result.timings['total'] < .5


async def test_total_deadline_cancels_ranking_and_preserves_retrieval():
    async def handle(request):
        if request.url.host == 'api.exa.ai':
            return httpx.Response(200, json={'results': [{'title': 'Paper', 'url': 'https://example.org/p', 'text': 'actual evidence'}]})
        await asyncio.sleep(10)
    async with Client(Config(api_keys={'exa': 'fixture', 'openrouter': 'fixture'}), transport=httpx.MockTransport(handle)) as client:
        result = await client.asearch('question', providers=['exa'], ranker='jev', deadline=.1, min_relevance=None)
    assert result.ranking_backend == 'rrf' and result.results and result.results[0].model_score is None
    assert any('returned RRF' in warning for warning in result.warnings)
    assert result.timings['total'] < .5


@pytest.mark.parametrize('kwargs', [{'deadline': 0}, {'deadline': float('nan')}, {'fast': 1}, {'fast_min_results': True}])
async def test_invalid_deadline_options(kwargs):
    async with Client(Config(api_keys={}), transport=httpx.MockTransport(lambda r: None)) as client:
        with pytest.raises(ValueError): await client.asearch('question', **kwargs, min_relevance=None)


async def test_graph_edge_direction_and_partial_failure():
    def paper(pid):
        return {'paperId': pid, 'title': pid, 'abstract': 'Source abstract', 'externalIds': {'DOI': '10.1/' + pid}}
    def handle(request):
        if request.url.path.endswith('/references'):
            return httpx.Response(200, json={'data': [{'citedPaper': paper('reference')}, {'citedPaper': None}], 'next': 2})
        if request.url.path.endswith('/citations'): return httpx.Response(403, json={'error': 'fixture-secret'})
        return httpx.Response(200, json=paper('seed'))
    async with Client(Config(api_keys={'semantic_scholar': 'fixture-secret'}), transport=httpx.MockTransport(handle)) as client:
        graph = await client.acitation_graph('DOI:10.1/seed', limit=2)
    assert graph['edges'] == [{'source': 'doi:10.1/seed', 'target': 'doi:10.1/reference', 'relation': 'cites', 'provider': 'semantic_scholar'}]
    assert graph['failed'] == {'citations': 'HTTPStatusError'}
    assert graph['truncated']['references'] and 'fixture-secret' not in str(graph)


async def test_usage_redacts_echoed_credentials_and_keeps_unknowns():
    def handle(request):
        if request.url.host == 'serpapi.com':
            return httpx.Response(200, json={'api_key': 'private-key', 'account_email': 'private', 'total_searches_left': 7})
        if request.url.host == 'api.firecrawl.dev': return httpx.Response(402, json={'error': 'private-key'})
        return httpx.Response(429, json={'error': 'quota exhausted private-key'})
    async with Client(Config(api_keys={p: 'private-key' for p in ['exa', 'serpapi', 'firecrawl', 'tavily']}), transport=httpx.MockTransport(handle)) as client:
        report = await client.ausage(providers=['exa', 'serpapi', 'firecrawl', 'tavily'])
    rows = {r['provider']: r for r in report['providers']}
    assert rows['exa']['status'] == 'unsupported' and rows['exa']['remaining'] is None
    assert rows['serpapi']['remaining'] == 7
    assert rows['tavily']['status'] == rows['firecrawl']['status'] == 'quota_exhausted'
    assert 'private' not in json.dumps(report)
