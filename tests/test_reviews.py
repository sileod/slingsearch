import json
import httpx
import pytest
from slingsearch import Client, Config, Claim, Result, novelty_search, load_papers
from slingsearch.review import report_tokens, plan_queries, normalize_claims


def response_paper(pid, title=None):
    return {'paperId': pid, 'title': title or pid, 'url': 'https://example.org/' + pid,
            'abstract': 'The source explicitly constructs verified proof trees and measures their required structure.',
            'year': 2024, 'externalIds': {'DOI': '10.1234/' + pid}}


async def test_seeded_review_excludes_known_variants_preserves_graph_trails_and_packing():
    calls = []
    def handle(request):
        calls.append(request)
        if request.url.host == 'api.exa.ai':
            return httpx.Response(200, json={'results': [
                {'title': 'Known', 'url': 'https://doi.org/10.1234/seed', 'text': 'Known proof methods'},
                {'title': 'New work', 'url': 'https://doi.org/10.1234/new', 'text': 'Source text describing proof trees.'}]})
        if request.url.host == 'openrouter.ai':
            body = json.loads(request.content)
            assert body['model'] == '~typesafe/jev-latest'
            assert body['state']['contribution'] == 'Certified proof structure'
            assert body['state']['document']
            answers = {}
            for key, value in body['questions'].items():
                assert value['instructions'].get('document') or body['state'].get('document')
                assert 'claim' in value['instructions']
                if value['type'] == 'choice':
                    assert set(value['criteria']) == {'same_contribution','partial_overlap','background','insufficient_evidence'}
                    answers[key] = {'choice':'partial_overlap','confidence':.8}
                else: answers[key] = {'noul':.6}
            return httpx.Response(200, json={'model':'typesafe/jev-1.14','answers':answers,'usage':{'cost':.001,'input_tokens':100}})
        if request.url.path.endswith('/batch'):
            return httpx.Response(200,json=[response_paper('seed','Known') for _ in json.loads(request.content)['ids']])
        if request.url.path.endswith('/references'):
            return httpx.Response(200, json={'data':[{'citedPaper': response_paper('neighbor')}]})
        if request.url.path.endswith('/citations'):
            return httpx.Response(200, json={'data':[{'citingPaper': response_paper('new')}]})
        return httpx.Response(200, json=response_paper('seed','Known'))
    async with Client(Config(api_keys={'exa':'fixture','openrouter':'fixture','semantic_scholar':'fixture'}), transport=httpx.MockTransport(handle)) as client:
        report = await client.arelated_work_search(['DOI:10.1234/seed'], contribution='Certified proof structure',
            claims=[Claim('structure','Certified structure'), Claim('transfer','Matched transfer')],
            providers=['exa'], max_queries=2, max_results=3, max_covered=1, token_budget=5000)
    assert report.covered and report.discoveries
    assert all(f.paper.doi != '10.1234/seed' for f in report.discoveries)
    neighbor = next(f for f in report.discoveries if f.paper.doi == '10.1234/neighbor')
    assert any(o['kind']=='graph' and o['direction']=='references' for o in neighbor.origins)
    new = next(f for f in report.discoveries if f.paper.doi == '10.1234/new')
    assert {o['kind'] for o in new.origins} == {'search','graph'}
    assert all(len(f.judgments)==2 for f in report.findings)
    assert all(j['relation']=='partial_overlap' for f in report.findings for j in f.judgments)
    assert len([r for r in calls if r.url.host=='openrouter.ai']) == 3  # Each paper shares its document across the two claim judgments.
    assert report.usage['overlap']['typed_questions'] == 12
    assert report.cost_usd == .003
    assert report_tokens(report) <= 5000
    assert '10.1234/neighbor' in report.to_markdown()


async def test_idea_only_missing_key_failure_and_budget_do_not_claim_novelty():
    def handle(request):
        return httpx.Response(200,json={'results':[{'title':'Some work','url':'https://example.org/p',
            'text':'Literal source evidence about proof construction. ' * 100}]})
    async with Client(Config(api_keys={'exa':'fixture'}),transport=httpx.MockTransport(handle)) as client:
        report = await client.anovelty_search('Certified proofs', claims=['Proof structure'],
            providers=['exa'], max_queries=2, ranker='jev', token_budget=450, expand_graph=False)
    assert report.ranking_backend=='rrf'
    assert report_tokens(report)<=450
    assert all(j['relation']=='unassessed' for f in report.findings for j in f.judgments)
    assert any('unavailable' in w for w in report.warnings)
    assert 'novelty_probability' not in report.to_dict()


async def test_paper_list_without_contribution_is_coverage_only():
    def handle(request):
        if request.url.host=='api.exa.ai':return httpx.Response(200,json={'results':[]})
        if request.url.path.endswith('/batch'):
            return httpx.Response(200,json=[response_paper('seed')])
        return httpx.Response(200,json=response_paper('seed'))
    async with Client(Config(api_keys={'exa':'fixture'}),transport=httpx.MockTransport(handle)) as client:
        report = await client.arelated_work_search(['ARXIV:2401.12345'],providers=['exa'],expand_graph=False,judge=None)
    assert not report.claims and report.judgment_model is None
    assert report.query_plan[0]['kind']=='seed_title'


def test_markdown_import_and_query_limits(tmp_path):
    path=tmp_path/'related.md'
    path.write_text('A [paper](https://arxiv.org/abs/2401.12345), [duplicate](https://arxiv.org/abs/2401.12345). [DOI](https://doi.org/10.1234/a).')
    seeds=load_papers(path)
    assert len(seeds)==2
    claims=normalize_claims('Idea', {'a':'First facet','b':'Second facet'})
    plan=plan_queries('Idea',claims,[],None,2)
    assert plan[0]['kind']=='combined_contribution'
    assert plan[0]['claim_ids']==['a','b'] and plan[1]['claim_ids']==['a']
    assert len(plan_queries('Idea',claims,[],['custom','custom'],10))==1


async def test_invalid_workflow_options_do_not_call_network():
    async with Client(Config(api_keys={}),transport=httpx.MockTransport(lambda request:pytest.fail('Unexpected network'))) as client:
        with pytest.raises(ValueError):await client.anovelty_search('')
        with pytest.raises(ValueError):await client.arelated_work_search([])
        with pytest.raises(ValueError):await client.anovelty_search('idea',max_queries=0)
        with pytest.raises(ValueError):await client.anovelty_search('idea',claims=[Claim('x','one'),Claim('x','two')])


async def test_failed_overlap_is_unassessed_not_background_or_novel():
    def handle(request):
        if request.url.host == 'api.exa.ai':
            return httpx.Response(200,json={'results':[{'title':'Paper','url':'https://example.org/p','text':'Actual supporting text.'}]})
        return httpx.Response(503,json={'error':'private-key'})
    async with Client(Config(api_keys={'exa':'private-key','openrouter':'private-key'}),transport=httpx.MockTransport(handle)) as client:
        report = await client.anovelty_search('Contribution',claims='One claim',queries='One query',
            providers=['exa'],expand_graph=False,token_budget=2000)
    assert len(report.query_plan)==1 and len(report.claims)==1
    assert report.findings[0].judgments[0]['relation']=='unassessed'
    assert report.findings[0].judgments[0]['status']=='unavailable'
    assert 'private-key' not in report.to_json()


async def test_unresolved_seed_alias_still_excludes_existing_paper():
    def handle(request):
        if request.url.host == 'api.exa.ai':
            return httpx.Response(200,json={'results':[{'title':'Known','url':'https://doi.org/10.1234/known','text':'Source text.'}]})
        return httpx.Response(429,json={'error':'rate limited'})
    async with Client(Config(api_keys={'exa':'fixture'}),transport=httpx.MockTransport(handle)) as client:
        report=await client.arelated_work_search('DOI:10.1234/known',contribution='Claim',queries=['query'],
            providers=['exa'],expand_graph=False,judge=None)
    assert report.unresolved_seeds and not report.discoveries and report.covered


def test_seeded_plan_prioritizes_known_neighborhood():
    seed=Result('s','Seed title','https://example.org/s')
    plan=plan_queries('Contribution',[Claim('c','Facet')],[seed],None,3,seeded=True)
    assert plan[0]['kind']=='seed_title'
    assert any(q['kind']=='combined_contribution' for q in plan)


async def test_review_keeps_substantive_excerpts_under_multiclaim_budget():
    source='The method enumerates minimal proofs, checks shortcuts and controls proof structure. ' * 50
    def handle(request):
        if request.url.host=='api.exa.ai':
            return httpx.Response(200,json={'results':[{'title':f'Paper {i}','url':f'https://example.org/{i}','text':source} for i in range(8)]})
        body=json.loads(request.content);answers={}
        for key,value in body['questions'].items():
            assert len(value['instructions'].get('document') or body['state']['document']) > 200
            answers[key]={'choice':'partial_overlap','confidence':.9} if value['type']=='choice' else {'noul':.3}
        return httpx.Response(200,json={'answers':answers})
    async with Client(Config(api_keys={'exa':'fixture','openrouter':'fixture'}),transport=httpx.MockTransport(handle)) as client:
        report=await client.anovelty_search('Structured proofs',claims=['Claim one','Claim two','Claim three','Claim four'],
            providers=['exa'],max_queries=1,max_results=8,token_budget=3000,expand_graph=False)
    assert 0 < len(report.findings) < 8
    assert report_tokens(report)<=3000
    assert all(sum(len(p) for p in f.paper.passages)>=250 for f in report.findings)
    assert any(j['relation']=='partial_overlap' for f in report.findings for j in f.judgments)
