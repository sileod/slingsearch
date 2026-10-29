import json
import httpx
import pytest
from slingsearch import Client, Config


def handler(calls, *, fail=False):
    def handle(request):
        if request.url.host == 'api.exa.ai':
            return httpx.Response(200, json={'results': [
                {'title':'A','url':'https://example.org/a','text':'Direct source evidence A'},
                {'title':'B','url':'https://example.org/b','text':'Related background B'}]})
        body = json.loads(request.content)
        calls.append(body)
        if request.url.path.endswith('/rerank'):
            return httpx.Response(200,json={'results':[{'index':1,'relevance_score':.99},{'index':0,'relevance_score':.1}]})
        if fail:
            return httpx.Response(503)
        answers={}
        for name,item in body['questions'].items():
            if item['type']=='score':
                document = item['instructions'].get('document','') if isinstance(item['instructions'],dict) else body['state']['document']
                answers[name]={'score':3 if 'evidence A' in document else 1}
            else:
                answers[name]={'noul':.9}
        return httpx.Response(200,json={'answers':answers,'model':'fixture-model','usage':{'cost':.001}})
    return handle


async def test_cutoff_reuses_decision_scores_and_does_not_fill_low_relevance_slots():
    calls=[]
    async with Client(Config(api_keys={'exa':'fixture','openrouter':'fixture'},decision_batch_size=5),transport=httpx.MockTransport(handler(calls))) as c:
        result=await c.asearch('question',providers=['exa'],min_relevance=.75)
    assert len(calls)==1
    assert [r.title for r in result.results]==['A']
    assert result.results[0].model_score==4
    assert result.results[0].relevance_score==.75
    assert result.usage['relevance_filter']['scored']==2
    assert result.usage['relevance_filter']['model']=='fixture-model'
    assert '1 qualified' in result.to_markdown()


async def test_voyage_cutoff_uses_common_rubric_not_native_score():
    calls=[]
    async with Client(Config(api_keys={'exa':'fixture','openrouter':'fixture'}),transport=httpx.MockTransport(handler(calls))) as c:
        result=await c.asearch('question',providers=['exa'],ranker='voyage',min_relevance=.75)
    assert len(calls)==2
    assert [r.title for r in result.results]==['A']
    assert result.results[0].model_score==.1
    assert result.results[0].relevance_score==.75
    assert result.usage['relevance_filter']['scores']['https://example.org/a']==.75
    assert result.ranking_backend=='voyage'


async def test_failure_cannot_turn_rrf_scores_into_qualified_results():
    calls=[]
    async with Client(Config(api_keys={'exa':'fixture','openrouter':'fixture'}),transport=httpx.MockTransport(handler(calls,fail=True))) as c:
        strict=await c.asearch('question',providers=['exa'],min_relevance=.75)
        normal=await c.asearch('question',providers=['exa'],min_relevance=None)
    assert strict.results==[]
    assert strict.usage['relevance_filter']['status']=='unavailable'
    assert 'not evidence' in strict.to_markdown()
    assert normal.results and normal.ranking_backend=='rrf'


@pytest.mark.parametrize('minimum',[-.1,1.1,True,3.5,float('nan'),float('inf'),'0.5'])
async def test_invalid_cutoffs_fail_before_network(minimum):
    async with Client(Config(api_keys={}),transport=httpx.MockTransport(lambda request: pytest.fail('unexpected network'))) as c:
        with pytest.raises(ValueError):
            await c.asearch('question',min_relevance=minimum)


async def test_review_cutoff_and_model_default():
    calls=[]
    async with Client(Config(api_keys={'exa':'fixture','openrouter':'fixture'},decision_batch_size=5),transport=httpx.MockTransport(handler(calls))) as c:
        result=await c.anovelty_search('question',providers=['exa'],expand_graph=False,judge=None,
                                      max_queries=1,min_relevance=.75)
    assert result.ranking_backend=='jev'
    assert [f.paper.title for f in result.findings]==['A']
    assert result.usage['relevance_filter']['kept']==1


async def test_default_cutoff_excludes_topical_background():
    calls=[]
    async with Client(Config(api_keys={'exa':'fixture','openrouter':'fixture'},decision_batch_size=5),transport=httpx.MockTransport(handler(calls))) as c:
        result=await c.asearch('question',providers=['exa'])
    assert [r.title for r in result.results]==['A']
    assert result.usage['relevance_filter']['minimum']==.5


async def test_default_without_scoring_key_is_explicitly_unavailable():
    calls=[]
    async with Client(Config(api_keys={'exa':'fixture'}),transport=httpx.MockTransport(handler(calls))) as c:
        result=await c.asearch('question',providers=['exa'])
    assert result.results==[] and result.successful==['exa']
    assert result.usage['relevance_filter']['status']=='unavailable'


@pytest.mark.parametrize('options,expected',[([],.5),(['--no-relevance-filter'],None),(['--min-relevance','0.75'],.75)])
def test_cli_threshold_defaults_and_explicit_opt_out(monkeypatch, options, expected, capsys):
    from slingsearch import cli
    from slingsearch.models import SearchResponse
    seen={}
    def search(**kwargs):
        seen.update(kwargs)
        return SearchResponse(kwargs['question'],kwargs['mode'])
    monkeypatch.setattr(cli,'search',search)
    monkeypatch.setattr('sys.argv',['slingsearch','question',*options])
    cli.main()
    assert seen['min_relevance']==expected
    assert seen['ranker']=='jev'


@pytest.mark.parametrize('cutoff,kept',[(0,['A','B']),(.25,['A','B']),(.2501,['A']),(.63,['A']),(.75,['A']),(.7501,[]),(1,[])])
async def test_normalized_cutoff_boundaries_and_arbitrary_floats(cutoff,kept):
    calls=[]
    async with Client(Config(api_keys={'exa':'fixture','openrouter':'fixture'},decision_batch_size=5),transport=httpx.MockTransport(handler(calls))) as c:
        result=await c.asearch('question',providers=['exa'],min_relevance=cutoff)
    assert [r.title for r in result.results]==kept
    assert result.usage['relevance_filter']['scale']=='normalized_0_to_1'
    assert result.usage['relevance_filter']['scores']=={'https://example.org/a':.75,'https://example.org/b':.25}
    assert result.usage['relevance_filter']['raw_scores']=={'https://example.org/a':4,'https://example.org/b':2}
