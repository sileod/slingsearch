import json
import httpx
import pytest
from slingsearch import Client, Config, Result
from slingsearch.providers.semantic_scholar import SemanticScholar
from slingsearch.fusion import fuse
from slingsearch.models import Source
from slingsearch.output import budget_results, evidence_tokens
from slingsearch.passages import prepare


def payload():
    return {'retrievalVersion':'fixture-v1','data':[
        {'paper':{'corpusId':'123','title':'Paper','authors':['Author']},
         'snippet':{'text':'Held out retrieval evaluation demonstrates cross domain generalization.', 'section':'Experiments','snippetKind':'body','snippetOffset':{'start':50,'end':120}}},
        {'paper':{'corpusId':'123','title':'Paper','authors':['Author']},
         'snippet':{'text':'The second original passage discusses limitations.', 'section':'Limitations','snippetKind':'body'}}]}


def handle(request):
    if request.url.path.endswith('/snippet/search'):return httpx.Response(200,json=payload())
    if request.url.path.endswith('/batch'):
        return httpx.Response(200,json=[{'paperId':'sha','corpusId':123,'title':'Paper','url':'https://doi.org/10.1234/paper',
            'abstract':'An abstract about unrelated introductory concepts.', 'externalIds':{'DOI':'10.1234/paper','CorpusId':123},'year':2024}])
    if request.url.host=='api.exa.ai':
        return httpx.Response(200,json={'results':[{'title':'Paper','url':'https://doi.org/10.1234/paper','text':'Original abstract-like excerpt.'}]})
    pytest.fail('Unexpected request')


async def test_snippet_contract_no_pagination_and_one_provider_vote():
    calls=[]
    def capture(request):calls.append(request);return handle(request)
    async with httpx.AsyncClient(transport=httpx.MockTransport(capture)) as client:
        adapter=SemanticScholar(client,'fixture','scholar',(2020,2025))
        rows=await adapter.search_snippets('The full original question?',10,paper_ids=['DOI:10.1234/paper'])
    assert len(rows)==2 and len(calls)==2
    params=calls[0].url.params
    assert params['query']=='The full original question?' and params['paperIds']=='DOI:10.1234/paper'
    assert params['year']=='2020-2025' and 'offset' not in params
    assert 'paper' not in params['fields']
    merged,=fuse(rows)
    assert merged.rrf_score==pytest.approx(1/61)
    assert merged.doi=='10.1234/paper' and merged.year==2024
    assert len(merged.passages)==2
    assert [(m['index'],m['section']) for m in merged.passage_metadata]==[(0,'Experiments'),(1,'Limitations')]


async def test_enrichment_preserves_votes_and_passages_despite_existing_abstract():
    async with Client(Config(api_keys={'exa':'fixture','semantic_scholar':'fixture'}),transport=httpx.MockTransport(handle)) as client:
        response=await client.asearch('Held out retrieval evaluation cross domain generalization',mode='scholar',
                        providers=['exa'],ranker='rrf',min_relevance=None,enrich_snippets=True,token_budget=2000)
    result,=response.results
    assert result.sources==[Source('exa',1)]
    assert result.rrf_score==pytest.approx(1/61)
    assert 'Held out retrieval' in result.passages[0]
    assert result.passage_metadata[0]['section']=='Experiments'
    assert response.usage['snippet_enrichment']['enriched_papers']==1
    assert 'Experiments' in response.to_markdown()
    assert evidence_tokens(response.results)<=2000


async def test_snippet_discovery_uses_existing_semantic_scholar_provider():
    async with Client(Config(api_keys={'semantic_scholar':'fixture'}),transport=httpx.MockTransport(handle)) as client:
        result=await client.asearch('retrieval evaluation',mode='scholar',providers=['semantic_scholar'],ranker='rrf',
                                   min_relevance=None,snippet_search=True,token_budget=2000)
    assert result.successful==['semantic_scholar']
    assert result.usage['semantic_scholar_retrieval']=='snippets'
    assert len(result.results)==1
    assert result.results[0].abstract is None  # Selected body text must not be labeled as an abstract.
    assert any('Held out retrieval' in p for p in result.results[0].passages)


async def test_enrichment_failure_keeps_existing_evidence():
    def fail(request):
        if request.url.host=='api.exa.ai':return handle(request)
        return httpx.Response(503)
    async with Client(Config(api_keys={'exa':'fixture'}),transport=httpx.MockTransport(fail)) as client:
        result=await client.asearch('question',mode='scholar',providers=['exa'],ranker='rrf',min_relevance=None,enrich_snippets=True)
    assert result.results[0].passages==['Original abstract-like excerpt.']
    assert result.usage['snippet_enrichment']['status']=='unavailable'


def test_selection_keeps_section_mapping_and_drops_stale_annotations():
    result=Result('p','Paper','https://example.org/p',passages=['Irrelevant original passage.','Held out retrieval evaluation evidence.'],
                  passage_metadata=[{'index':0,'section':'Introduction'},{'index':1,'section':'Experiments'}],
                  abstract='General abstract text.')
    prepare(result,'Held out retrieval evaluation',1)
    assert result.passage_metadata==[{'index':0,'section':'Experiments'}]
    bounded=budget_results([result],1,500)
    assert bounded[0].passage_metadata[0]['index']==0


async def test_unmatched_restricted_result_cannot_attach_to_another_paper():
    def other(request):
        if request.url.host=='api.exa.ai':return httpx.Response(200,json={'results':[{'title':'Paper','url':'https://doi.org/10.1234/other','text':'Unchanged source.'}]})
        return handle(request)
    async with Client(Config(api_keys={'exa':'fixture'}),transport=httpx.MockTransport(other)) as client:
        result=await client.asearch('question',mode='scholar',providers=['exa'],ranker='rrf',min_relevance=None,enrich_snippets=True)
    assert result.results[0].passages==['Unchanged source.']
    assert result.usage['snippet_enrichment']['enriched_papers']==0


async def test_invalid_snippet_options_fail_before_network():
    async with Client(Config(api_keys={}),transport=httpx.MockTransport(lambda r:pytest.fail('network'))) as c:
        with pytest.raises(ValueError):await c.asearch('q',snippet_search=True)
        with pytest.raises(ValueError):await c.asearch('q',mode='scholar',enrich_snippets='yes')


async def test_metadata_failure_retains_corpus_identity_and_section():
    def failure(request):
        if request.url.path.endswith('/snippet/search'):return httpx.Response(200,json=payload())
        return httpx.Response(503)
    async with httpx.AsyncClient(transport=httpx.MockTransport(failure)) as client:
        adapter=SemanticScholar(client,'fixture','scholar')
        rows=await adapter.search_snippets('question',2)
    assert rows[0].identifiers=={'corpusid':'123'}
    assert rows[0].passage_metadata[0]['section']=='Experiments'
    assert adapter.snippet_warnings


async def test_shared_client_serializes_s2_requests():
    active=0;peak=0
    async def slow(request):
        nonlocal active,peak
        import asyncio
        active+=1;peak=max(peak,active)
        await asyncio.sleep(.01)
        active-=1
        return httpx.Response(200,json={'data':[]})
    import asyncio
    async with httpx.AsyncClient(transport=httpx.MockTransport(slow)) as client:
        a,b=SemanticScholar(client),SemanticScholar(client)
        await asyncio.gather(a.search('first'),b.search_snippets('second'))
    assert peak==1


async def test_literal_asterisk_400_retries_once_without_changing_input_question():
    queries=[]
    def response(request):
        query=request.url.params['query'];queries.append(query)
        return httpx.Response(400,json={'error':'Unsupported query'}) if '*' in query else httpx.Response(200,json={'data':[]})
    async with httpx.AsyncClient(transport=httpx.MockTransport(response)) as client:
        adapter=SemanticScholar(client,'fixture','scholar')
        question='How does A* select nodes?'
        assert await adapter.search_snippets(question)==[]
    assert queries==[question,question.replace('*',' ')]
    assert adapter.snippet_warnings


def test_stable_s2_url_resolves_enrichment_identity_without_title_matching():
    from slingsearch.fusion import normalize
    from slingsearch.snippets import paper_identifier
    sha='a'*40
    result=normalize(Result('p','Paper','https://www.semanticscholar.org/paper/paper-title/'+sha))
    assert paper_identifier(result)==sha


def test_output_trimming_does_not_misattribute_shared_prefixes():
    prefix='Identical introduction sentence. '
    result=Result('p','Paper','https://example.org/p',passages=[prefix+'A '*200,prefix+'B '*200],
                  passage_metadata=[{'index':0,'section':'Methods'},{'index':1,'section':'Results'}])
    final=budget_results([result],1,300)
    assert final and evidence_tokens(final)<=300
    assert len(final[0].passage_metadata)==len(final[0].passages)
    assert [m['section'] for m in final[0].passage_metadata]==['Methods','Results'][:len(final[0].passages)]


def test_selected_body_excerpt_is_never_mislabeled_as_abstract():
    result=Result('p','Paper','https://example.org/p',abstract='Generic introductory abstract.',
                  passages=['Held out retrieval evaluation evidence from the body.'],
                  passage_metadata=[{'index':0,'section':'Experiments','snippet_kind':'body'}])
    prepare(result,'Held out retrieval evaluation',1)
    bounded=budget_results([result],1,600)
    assert bounded[0].abstract is None
    assert bounded[0].passages==result.passages
    assert bounded[0].passage_metadata[0]['snippet_kind']=='body'
