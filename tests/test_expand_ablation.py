import json
import httpx
import pytest
from benchmarks.datasets import Query
from slingsearch import Result
from slingsearch.models import Source
from benchmarks.expand_ablation import candidates_for,evidence_key
from benchmarks.span_free import score_documents


def test_ablation_arms_are_isolated_and_do_not_modify_frozen_results():
    q=Query('q','question',candidates=[Result('a','Base','https://example.org/a',passages=['Base evidence'],sources=[Source('exa',1)]),
                                    Result('b','Added','https://example.org/b',passages=['Added evidence'],sources=[Source('expand',1)])])
    baseline,_=candidates_for(q,'default');expanded,_=candidates_for(q,'default+expand');single,_=candidates_for(q,'expand')
    assert [r.title for r in baseline]==['Base'] and len(expanded)==2
    assert [r.title for r in single]==['Added']
    assert q.candidates[0].rrf_score==0 and q.candidates[0].passages==['Base evidence']


def test_blind_evidence_identity_distinguishes_different_displayed_excerpts():
    paper={'id':'p','title':'Paper','passages':['A']}
    assert evidence_key('q',paper)!=evidence_key('q',{**paper,'passages':['B']})


async def test_free_noul_scoring_validates_mapping_and_reuses_identical_inputs(tmp_path):
    calls=[]
    def handle(request):
        payload=json.loads(request.content);calls.append(payload)
        assert payload['model']=='respan/span-01-lite:free'
        assert all(q['type']=='noul' for q in payload['questions'].values())
        return httpx.Response(200,json={'model':'fixture','answers':{name:{'noul':.8} for name in payload['questions']},'usage':{'cost':0}})
    docs=[Result('a','A','https://example.org/a',passages=['Evidence'])]
    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        scores,usage=await score_documents(client,'fixture','question',docs,tmp_path/'scores.json',pace_seconds=0)
        cached,_=await score_documents(client,'fixture','question',docs,tmp_path/'scores.json',pace_seconds=0)
    assert len(calls)==1 and cached==scores and usage['reported_cost_usd']==0
    assert next(iter(scores.values()))['score']==.8
    assert 'fixture' not in json.dumps(calls)


def test_scoring_identity_resolution_does_not_inject_target_text_or_mutate_inputs():
    from benchmarks.expand_report import resolve_for_scoring
    from benchmarks.academic_eval import lookup_id
    source=Result('x','Original title','https://arxiv.org/abs/2401.12345',passages=['Original provider wording'])
    metadata={lookup_id(source):{'paperId':'sha','corpusId':123,'abstract':'Target text must not enter ranking','externalIds':{'DOI':'10.1234/id'}}}
    resolved,=resolve_for_scoring([source],metadata)
    assert resolved.doi=='10.1234/id'
    assert resolved.passages==source.passages and resolved.abstract is None
    assert source.identifiers=={} and source.doi is None
