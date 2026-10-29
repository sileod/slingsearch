import json
import httpx
import pytest
from slingsearch import Client, Config


async def test_all_checks_required_without_reordering_or_native_score_replacement():
    seen=[]
    def handle(request):
        if request.url.host=='api.exa.ai':
            return httpx.Response(200,json={'results':[
                {'title':name,'url':'https://example.org/'+name,'text':'Original excerpt '+name} for name in ['A','B']]})
        body=json.loads(request.content);seen.append(body)
        answers={}
        for name,q in body['questions'].items():
            assert q['type']=='noul'
            assert body['state']=={'question':'question'}
            text=q['instructions']['document']
            check=q['instructions']['requirement']
            answers[name]={'noul': .2 if 'excerpt A' in text and check=='Second requirement?' else .9}
        return httpx.Response(200,json={'model':'fixture','answers':answers,'usage':{'cost':.002}})
    async with Client(Config(api_keys={'exa':'fixture','openrouter':'fixture'}),transport=httpx.MockTransport(handle)) as c:
        response=await c.asearch('question',providers=['exa'],ranker='rrf',min_relevance=None,
                                checks=['First requirement?','Second requirement?'])
    assert len(seen)==1 and len(seen[0]['questions'])==4
    assert [r.title for r in response.results]==['B']
    assert response.results[0].model_score is None
    assert response.usage['checks']['results'][0]['scores']==[.9,.2]
    assert response.cost_usd==.002
    assert '1 passed all checks' in response.to_markdown()


async def test_missing_key_does_not_pass_required_checks():
    def handle(request):
        return httpx.Response(200,json={'results':[{'title':'A','url':'https://example.org/a','text':'Source excerpt'}]})
    async with Client(Config(api_keys={'exa':'fixture'}),transport=httpx.MockTransport(handle)) as c:
        response=await c.asearch('question',providers=['exa'],ranker='rrf',min_relevance=None,checks=['Evidence?'])
    assert response.results==[]
    assert response.usage['checks']['status']=='unavailable'


@pytest.mark.parametrize('checks,threshold',[('not a list',.7),([''],.7),(['x']*11,.7),(['x'],float('nan')),(['x'],0)])
async def test_check_validation_precedes_network(checks,threshold):
    async with Client(Config(api_keys={}),transport=httpx.MockTransport(lambda r:pytest.fail('network'))) as c:
        with pytest.raises(ValueError):await c.asearch('question',checks=checks,check_threshold=threshold)


async def test_incomplete_check_mapping_fails_closed():
    def handle(request):
        if request.url.host=='api.exa.ai':
            return httpx.Response(200,json={'results':[{'title':'A','url':'https://example.org/a','text':'Source excerpt'}]})
        return httpx.Response(200,json={'answers':{},'usage':{'cost':.001}})
    async with Client(Config(api_keys={'exa':'fixture','openrouter':'fixture'}),transport=httpx.MockTransport(handle)) as c:
        result=await c.asearch('question',providers=['exa'],ranker='rrf',min_relevance=None,checks=['Evidence?'])
    assert result.results==[] and result.usage['checks']['status']=='unavailable'
    assert result.cost_usd==.001
