import json
import math
import httpx
import pytest
from benchmarks.datasets import doris, paperfinding
from benchmarks.scholar_eval import evaluate, lexical_order, SpendGuard
from benchmarks.paperfinding_eval import parse_judgment, metadata_id
from slingsearch import Result, Source
from slingsearch.fusion import fuse


def native(tmp_path):
    payload = {'aspect_id2aspect':{'a':'Uses transformers','b':'Handles time series'},
               'Query':[{'query_text':'transformers for time series','candidate_pool':[0,1],
                         'aspects':{'a':['b']}}],
               'Corpus':[{'abstract_id':0,'title':'One','original_abstract':'transformers for time series',
                          'url':'https://arxiv.org/abs/1'},
                         {'abstract_id':1,'title':'Two','original_abstract':'unrelated discussion',
                          'url':'https://arxiv.org/abs/2'}],
               'Annotation':[{'aspect_id':a,'abstract_id':pid,'score':score}
                             for a,pid,score in [('a',0,2),('b',0,2),('a',1,1),('b',1,0)]]}
    p=tmp_path/'native.json';p.write_text(json.dumps(payload));return p,payload


def test_doris_native_gains_and_coverage(tmp_path):
    path,_=native(tmp_path)
    case=doris.load(path,[0])[0]
    assert case.query.qrels=={'doris:0':4,'doris:1':1}
    assert evaluate(case,['doris:0','doris:1'])['nDCG@5']==1
    assert evaluate(case,['doris:1'])['aspect_coverage@5']==.25
    assert evaluate(case,['doris:1'])['R@10']==0
    assert evaluate(case,['doris:1','doris:0'])['nDCG@5']==pytest.approx(
        (1+4/math.log2(3))/(4+1/math.log2(3)))
    assert lexical_order(case.query.question,list(reversed(case.query.candidates)))[0].id=='doris:0'
    with pytest.raises(ValueError):evaluate(case,['doris:0','doris:0'])
    with pytest.raises(ValueError):doris.load(path,[0,0])
    frozen=tmp_path/'frozen.json';doris.save_cases([case],frozen)
    assert doris.load_cases(frozen)[0]==case


def test_doris_rejects_incomplete_or_unresolved_judgments(tmp_path):
    path,data=native(tmp_path)
    data['Annotation'].pop();path.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='Incomplete'):doris.load(path,[0])
    data['Annotation'][0]['score']=-1;path.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='unresolved'):doris.load(path,[0])


def test_doris_only_checks_judgments_needed_by_each_query(tmp_path):
    path,data=native(tmp_path)
    data['aspect_id2aspect']['c']='Other requirement'
    data['Corpus'].append({'abstract_id':2,'title':'Third','original_abstract':'Other evidence',
                           'url':'https://arxiv.org/abs/3'})
    data['Query'].append({'query_text':'Other question','candidate_pool':[2],'aspects':{'c':[]}})
    data['Annotation'].extend([{'aspect_id':'c','abstract_id':pid,'score':score}
                               for pid,score in [(2,2),(0,0),(0,1)]])
    path.write_text(json.dumps(data))
    assert len(doris.load(path,[0,1]))==2  # c/0 is not in either question's evaluated pairs.
    data['Annotation'].append({'aspect_id':'a','abstract_id':0,'score':0})
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='Conflicting'):doris.load(path,[0,1])


def test_paperfinding_does_not_treat_known_positives_as_exhaustive(tmp_path):
    p=tmp_path/'pf.json'
    p.write_text(json.dumps([
        {'input':{'query_id':'semantic_1','query':'Find papers'},'scorer_criteria':{
            'known_to_be_good':['42'],'relevance_criteria':[{'name':'task','description':'Matches task','weight':1}]}},
        {'input':{'query_id':'specific_1','query':'Find BERT'},'scorer_criteria':{'corpus_ids':['43']}}]))
    broad,specific=paperfinding.load(p)
    assert not broad.query.qrels
    assert specific.query.qrels=={'corpusid:43':1}
    assert broad.criteria['known_to_be_good']==['42']


async def test_spend_guard_reserves_before_sending_and_persists(tmp_path):
    p=tmp_path/'spend.json'
    guard=SpendGuard(p,.0002)
    await guard.inner.aclose()
    guard.inner=httpx.MockTransport(lambda request:httpx.Response(200,json={'usage':{'cost':.00001}}))
    async with httpx.AsyncClient(transport=guard) as client:
        from slingsearch.config import MODELS
        body={'model':MODELS['jev']}
        response=await client.post('https://openrouter.ai/api/alpha/decisions',json=body)
        assert response.is_success
        # A replay is free and must not consume the reservation again.
        assert (await client.post('https://openrouter.ai/api/alpha/decisions',json=body)).is_success
        with pytest.raises(ValueError,match='reservation'):
            await client.post('https://openrouter.ai/api/alpha/decisions',json={**body,'state':{'question':'different'}})
    assert json.loads(p.read_text())['requests']==1
    assert SpendGuard(p,.0002).state['reported_usd']==.00001


async def test_full_sample_runner_and_free_replay(tmp_path,monkeypatch):
    from benchmarks import scholar_eval
    import hashlib
    path,_=native(tmp_path)
    cases=doris.load(path,[0])
    out=tmp_path/'run';out.mkdir()
    doris.save_cases(cases,out/'frozen.json')
    manifest={'snapshot_sha256':hashlib.sha256((out/'frozen.json').read_bytes()).hexdigest(),
              'ranking_reservation_limit_usd':.01,'seed':1,
              'methods':['bm25','random','jev','decider','voyage']}
    (out/'manifest.json').write_text(json.dumps(manifest))
    class Ranker:
        resolved_models={'fixture'}
        usage={}
        cost_usd=0
        async def rank(self,question,candidates):
            return candidates
    monkeypatch.setattr(scholar_eval,'make_ranker',lambda *args:Ranker())
    await scholar_eval.run(out,'not-a-real-key')
    first=json.loads((out/'results.json').read_text())
    assert first['summary']['jev']['successful']==1
    assert first['summary']['jev']['output_tokens_mean']<=1500
    assert len(first['paired_query_ids'])==1
    await scholar_eval.run(out,'not-a-real-key')
    assert len((out/'rankings.jsonl').read_text().splitlines())==5
    assert json.loads((out/'results.json').read_text())==first


def test_acl_pdf_and_doi_share_document_identity():
    docs=fuse([
        Result('a','Strategies for framing','https://aclanthology.org/2022.inlg-main.20.pdf',sources=[Source('exa',1)]),
        Result('b','Strategies for Framing','https://doi.org/10.18653/v1/2022.inlg-main.20',sources=[Source('openalex',2)])])
    assert len(docs)==1
    assert docs[0].identifiers['acl']=='2022.inlg-main.20'
    assert len(docs[0].sources)==2
    assert metadata_id(docs[0])=='DOI:10.18653/v1/2022.inlg-main.20'


def test_paperfinding_judge_names_and_missing_criteria():
    criteria=[{'name':'Distributed RL','weight':.75},{'name':'Decoupled workers','weight':.25}]
    payload={'criteria':{'distributed_rl':{'relevance':'Perfectly Relevant'},
                         'decoupled workers':{'relevance':'Somewhat Relevant'}}}
    judgment=parse_judgment(payload,criteria)
    assert judgment['grade']==2
    assert judgment['criterion_scores']=={'Distributed RL':3,'Decoupled workers':1}
    del payload['criteria']['distributed_rl']
    with pytest.raises(ValueError,match='Missing'):parse_judgment(payload,criteria)


async def test_paperfinding_rank_judge_and_report_replay(tmp_path,monkeypatch):
    from benchmarks import paperfinding_eval as pf
    from benchmarks.datasets import Query
    from dataclasses import asdict
    out=tmp_path/'pf';out.mkdir()
    docs=[Result('a','A','https://example.org/a',['Evidence for task'],year=2024,
                 sources=[Source('exa',1)]),
          Result('b','B','https://example.org/b',['Other task evidence'],year=2024,
                 sources=[Source('openalex',1)])]
    q=Query('semantic_1','Find papers on the task','scholar',candidates=docs)
    (out/'frozen.jsonl').write_text(json.dumps(asdict(q))+'\n')
    (out/'criteria.json').write_text(json.dumps({'semantic_1':{'known_to_be_good':[],
        'relevance_criteria':[{'name':'Task','description':'Discusses task','weight':1}]}}))
    class Ranker:
        resolved_models={'fixture'}
        usage={}
        cost_usd=0
        async def rank(self,question,candidates):return candidates
    monkeypatch.setattr(pf,'make_ranker',lambda *args:Ranker())
    await pf.rank(out,'fixture-secret')
    source=tmp_path/'source.py'
    source.write_text('relevance_criteria_judgement_prompt_with_relevant_snippets_after = "Judge {criteria}"')
    actual_client=httpx.AsyncClient
    calls=[]
    def handle(request):
        calls.append(request)
        assert request.url.host=='openrouter.ai'
        return httpx.Response(200,json={'model':'openai/gpt-4o-2024-11-20',
            'usage':{'prompt_tokens':100,'completion_tokens':20,'cost':.00045},
            'choices':[{'message':{'content':json.dumps({'criteria':{
                'task':{'relevance':'Perfectly Relevant','relevant_snippet':'task'}}})}}]})
    def fixture_client(**kwargs):
        kwargs['transport']=httpx.MockTransport(handle)
        return actual_client(**kwargs)
    monkeypatch.setattr(pf.httpx,'AsyncClient',fixture_client)
    await pf.judge(out,'fixture-secret',source)
    await pf.judge(out,'fixture-secret',source)
    assert len(calls)==2  # One judgment per unique paper; replay makes no requests.
    reference=tmp_path/'normalizer.json';reference.write_text(json.dumps({'semantic_1':2}))
    pf.report(out,reference)
    summary=json.loads((out/'results.json').read_text())['summary']
    assert summary['single:exa']['P@5_grade2plus']==.2
    assert summary['voyage']['P@5_grade2plus']==.4
    assert summary['voyage']['useful_non_exa_count@5']==1
    assert 'fixture-secret' not in ''.join(p.read_text() for p in out.glob('*') if p.is_file())
