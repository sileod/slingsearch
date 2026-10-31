import json
import httpx
import pytest
from slingsearch import Client,Config
from slingsearch.models import Source
from slingsearch.providers.you import You


async def test_you_post_highlights_contract_and_no_publication_year_inference():
    def handle(request):
        assert str(request.url)=='https://ydc-index.io/v1/search' and request.method=='POST'
        assert request.headers['X-API-Key']=='fixture'
        assert json.loads(request.content)=={'query':'Full question?','count':3,'extraction':{'extraction_mode':'highlights'}}
        return httpx.Response(200,json={'results':{'web':[{'url':'https://example.org/p','title':'Paper',
            'page_age':'2025-01-01','contents':{'highlights':['Original evidence.','Other evidence.']}}]}})
    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as c:
        a=You(c,'fixture','scholar');r,=await a.search('Full question?',3)
    assert r.passages==['Original evidence.','Other evidence.']
    assert r.sources==[Source('you',1)] and r.year is None and r.model_score is None
    assert r.passage_metadata[0]['snippet_kind']=='highlight'
    assert a.cost_usd is None


async def test_you_snippet_fallback_and_empty_results():
    replies=iter([{'results':{'web':[{'url':'https://example.org/','snippets':['Exact fallback.']}]}},{'results':{'web':[]}}])
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r:httpx.Response(200,json=next(replies)))) as c:
        a=You(c,'fixture');r,=await a.search('q');assert r.passages==['Exact fallback.']
        assert r.passage_metadata[0]['snippet_kind']=='search_excerpt'
        assert await a.search('q')==[]


async def test_you_selectable_and_key_missing_is_skipped():
    def handle(request):return httpx.Response(200,json={'results':{'web':[{'url':'https://example.org/','title':'A','snippets':['Evidence.']}]}})
    async with Client(Config(api_keys={'you':'fixture'}),transport=httpx.MockTransport(handle)) as c:
        r=await c.asearch('q',providers=['you'],ranker='rrf',min_relevance=None)
    assert r.successful==['you'] and r.results[0].sources==[Source('you',1)]
    async with Client(Config(api_keys={}),transport=httpx.MockTransport(lambda r:pytest.fail('network'))) as c:
        r=await c.asearch('q',providers=['you'],ranker='rrf',min_relevance=None)
    assert r.skipped=={'you':'Missing YOU_API_KEY'}


@pytest.mark.parametrize('query,limit',[('',1),('q',0),('q',51),('q',True)])
async def test_you_validation(query,limit):
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r:pytest.fail('network'))) as c:
        with pytest.raises(ValueError):await You(c).search(query,limit)


def test_you_key_alias_and_precedence(monkeypatch):
    monkeypatch.delenv('YOU_API_KEY',raising=False);monkeypatch.setenv('YDC_API_KEY','alias-fixture')
    assert Config().api_keys['you']=='alias-fixture'
    monkeypatch.setenv('YOU_API_KEY','preferred-fixture')
    config=Config();assert config.api_keys['you']=='preferred-fixture'
    assert 'you' not in config.web_providers and 'you' not in config.scholar_providers
