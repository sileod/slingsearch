import httpx
import pytest
from slingsearch import Client, Config
from slingsearch.providers.expand import Expand
from slingsearch.models import Source


async def test_expand_contract_and_native_scores_are_not_model_scores():
    import json
    calls=[]
    def handle(request):
        calls.append(request)
        assert request.method=='POST' and str(request.url)=='https://api.expand.ai/v1/search'
        assert request.headers['x-expand-api-key']=='fixture'
        assert json.loads(request.content)=={'query':'Full original question?','maxResults':10}
        assert b'fixture' not in request.content
        return httpx.Response(200,json={'id':'search-id','query':'Full original question?','results':[
            {'title':'Paper','url':'https://doi.org/10.1234/test','excerpt':'Original source excerpt.','score':.99}]})
    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        adapter=Expand(client,'fixture','scholar')
        row,=await adapter.search('Full original question?',50)
    assert len(calls)==1
    assert row.passages==['Original source excerpt.']
    assert row.sources==[Source('expand',1)] and row.model_score is None
    assert row.passage_metadata[0]['snippet_kind']=='search_excerpt'
    assert adapter.cost_usd is None


async def test_expand_is_selectable_and_missing_key_is_skipped():
    def handle(request):
        return httpx.Response(200,json={'results':[{'url':'https://example.org/p','title':'A','excerpt':'Evidence.'}]})
    async with Client(Config(api_keys={'expand':'fixture'}),transport=httpx.MockTransport(handle)) as c:
        result=await c.asearch('question',providers=['expand'],ranker='rrf',min_relevance=None)
    assert result.successful==['expand'] and result.results[0].sources==[Source('expand',1)]
    async with Client(Config(api_keys={}),transport=httpx.MockTransport(lambda r:pytest.fail('network'))) as c:
        result=await c.asearch('question',providers=['expand'],ranker='rrf',min_relevance=None)
    assert result.skipped=={'expand':'Missing EXPAND_API_KEY'}


async def test_expand_empty_results_and_missing_excerpt_do_not_fabricate_passages():
    replies=iter([{'results':[]},{'results':[{'url':'https://example.org/','excerpt':'','title':''}]}])
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r:httpx.Response(200,json=next(replies)))) as c:
        a=Expand(c,'fixture')
        assert await a.search('question')==[]
        row,=await a.search('question')
    assert row.title=='' and row.passages==[] and row.passage_metadata==[]


@pytest.mark.parametrize('question,limit',[('',1),('x'*2001,1),('q',0),('q',True)])
async def test_expand_validation_before_network(question,limit):
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r:pytest.fail('network'))) as c:
        with pytest.raises(ValueError):await Expand(c).search(question,limit)


async def test_expand_failure_is_partial_and_never_serializes_key():
    def handle(request):
        if request.url.host=='api.expand.ai':return httpx.Response(401,json={'message':'fixture-secret'})
        return httpx.Response(200,json={'results':[{'url':'https://example.org/','title':'A','text':'Original excerpt.'}]})
    async with Client(Config(api_keys={'expand':'fixture-secret','exa':'fixture'}),transport=httpx.MockTransport(handle)) as c:
        result=await c.asearch('question',providers=['expand','exa'],ranker='rrf',min_relevance=None)
    assert result.failed=={'expand':'HTTPStatusError'} and result.results
    assert 'fixture-secret' not in result.to_json()


def test_expand_environment_key_is_discovered_without_changing_defaults(monkeypatch):
    monkeypatch.setenv('EXPAND_API_KEY','fixture')
    config=Config()
    assert config.api_keys['expand']=='fixture'
    assert 'expand' not in config.web_providers and 'expand' not in config.scholar_providers
