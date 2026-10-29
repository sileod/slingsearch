import json
import httpx
import pytest
from slingsearch import Config, Result
from slingsearch.rankers import make_ranker


async def test_packing_preserves_independent_documents_and_cache_metrics():
    requests=[]
    def handle(request):
        body=json.loads(request.content);requests.append(body)
        assert body['state']=={'question':'Which evidence helps?'}
        answers={}
        for key,q in body['questions'].items():
            assert 'document' in q['instructions']
            assert q['instructions']['document'].startswith(('Paper A','Paper B','Paper C'))
            answers[key]={'score':3,'confidence':.8,'probabilities':{'3':1}} if q['type']=='score' else {'noul':.9}
        return httpx.Response(200,json={'answers':answers,'usage':{
            'input_tokens':100,'input_tokens_details':{'cached_tokens':20,'cache_write_tokens':10},'cost':.001}})
    config=Config(api_keys={'openrouter':'fixture'},decision_batch_size=2,decision_evidence_checks=True)
    docs=[Result(x,'Paper '+x,'https://example.org/'+x,['source text']) for x in 'ABC']
    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        ranker=make_ranker('jev',client,config);ranked=await ranker.rank('Which evidence helps?',docs)
    assert len(requests)==2
    assert [len(r['questions']) for r in requests]==[6,3]
    assert ranker.usage['ranking_requests']==2 and ranker.usage['decision_questions']==9
    assert ranker.usage['cached_tokens']==40 and ranker.usage['cache_write_tokens']==20
    assert ranker.usage['input_tokens']==200  # Cache reads are included, never added twice.
    assert all(r.model_score==4 for r in ranked) and all(r.model_score is None for r in docs)
    assert all(d['needs_more_text']==.9 for d in ranker.judgments.values())


async def test_packed_missing_mapping_fails_whole_ranking():
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r:httpx.Response(
            200,json={'answers':{'doc_0':{'score':4}}}))) as client:
        ranker=make_ranker('jev',client,Config(api_keys={'openrouter':'fixture'},decision_batch_size=2))
        docs=[Result(x,x,'https://example.org/'+x) for x in 'AB']
        with pytest.raises(ValueError,match='judgments failed'):await ranker.rank('q',docs)
    assert not ranker.judgments and all(r.model_score is None for r in docs)


@pytest.mark.parametrize('value',[0,21,True,1.5])
def test_batch_size_rejected_early(value):
    with pytest.raises(ValueError,match='decision_batch_size'):Config(decision_batch_size=value)


def test_global_budget_survives_restarts_and_enforces_inflight_reservations(tmp_path):
    from benchmarks.decision_budget import Budget
    first=Budget(tmp_path)
    token=first.reserve('fixture',2.)
    restarted=Budget(tmp_path)
    with pytest.raises(ValueError,match='budget'):restarted.reserve('fixture',.6)
    restarted.settle(token,.1)
    second=restarted.reserve('fixture',2.4)
    with pytest.raises(ValueError,match='budget'):first.reserve('fixture',.001)
    first.settle(second,.2)
    data=json.loads(first.path.read_text())
    assert data['charged_usd']==pytest.approx(.3) and not data['outstanding']


async def test_private_cache_replay_does_not_charge_again(tmp_path):
    from benchmarks.decision_budget import Transport
    transport=Transport(tmp_path);await transport.inner.aclose()
    calls=[]
    def handle(request):
        calls.append(request)
        return httpx.Response(200,json={'answers':{'relevance':{'score':3}},'usage':{'cost':.00001}})
    transport.inner=httpx.MockTransport(handle)
    async with httpx.AsyncClient(transport=transport) as client:
        for _ in range(2):
            response=await client.post('https://openrouter.ai/api/alpha/decisions',
                headers={'Authorization':'Bearer fixture-private-secret'},
                json={'model':'typesafe/jev-1.14','state':'document','questions':{'relevance':{'type':'score'}}})
            assert response.is_success
    data=json.loads((tmp_path/'api-budget.json').read_text())
    assert len(calls)==1 and data['charged_usd']==.00001 and data['cache_hits']==1
    assert 'fixture-private-secret' not in ''.join(p.read_text() for p in tmp_path.rglob('*.json'))
