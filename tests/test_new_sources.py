import httpx
import pytest
from slingsearch import Client, Config, Result
from slingsearch.providers import PROVIDERS
from slingsearch.fusion import fuse
from slingsearch.output import budget_results, evidence_tokens


async def test_github_unauthenticated_repository_contract():
    def handle(request):
        assert request.url.params['q'] == 'diffusion'
        assert 'Authorization' not in request.headers
        return httpx.Response(200, json={'items': [{'id': 1, 'full_name': 'org/repo', 'html_url': 'https://github.com/org/repo',
            'description': 'Original repository description', 'owner': {'login': 'org'}}]})
    async with Client(Config(api_keys={}), transport=httpx.MockTransport(handle)) as client:
        response = await client.asearch('diffusion', providers=['github'], ranker='rrf')
    assert response.successful == ['github'] and response.results[0].kind == 'repository'
    assert response.results[0].code_urls == ['https://github.com/org/repo']


async def test_pwc_current_backend_and_linked_metadata():
    def handle(request):
        assert request.url.host == 'paperswithcode.co'
        if request.url.path.endswith('/search'):
            assert request.url.params['mode'] == 'keyword'
            return httpx.Response(200, json={'results': [{'id': '1', 'arxiv_id': '2501.00001'}], 'next_page': None})
        assert request.url.params['include_resources'] == 'true'
        return httpx.Response(200, json={'id': '1', 'arxiv_id': '2501.00001', 'title': 'Paper', 'published': '2025-01-01',
            'abstract': 'Source abstract', 'authors': ['Author'], 'url_abs': 'https://arxiv.org/abs/2501.00001',
            'repositories': [{'url': 'https://github.com/org/repo'}], 'tasks': [{'name': 'Task'}],
            'hf_datasets': ['https://huggingface.co/datasets/org/data'], 'tldr': 'Generated text must not be used'})
    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        docs = await PROVIDERS['papers_with_code'](client, mode='scholar').search('question', 1)
    assert docs[0].code_urls and docs[0].tasks == ['Task'] and docs[0].abstract == 'Source abstract'
    assert docs[0].kind == 'paper'
    assert 'Generated text' not in str(docs)


async def test_huggingface_abstract_and_identifier():
    def handle(request):
        assert request.url.path == '/api/papers/search' and len(request.url.params['q']) <= 250
        return httpx.Response(200, json=[{'paper': {'id': '2501.00001', 'title': 'Paper', 'summary': 'Actual abstract',
            'authors': [{'name': 'Author'}], 'publishedAt': '2025-01-01T00:00:00Z', 'githubRepo': 'https://github.com/org/repo',
            'ai_summary': 'Do not use'}}])
    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        doc, = await PROVIDERS['huggingface'](client, mode='scholar').search('q' * 300)
    assert doc.abstract == 'Actual abstract' and doc.identifiers['arxiv'] == '2501.00001'
    assert doc.kind == 'paper'
    assert 'Do not use' not in str(doc)


def test_paper_repository_identity_and_budgeted_link_presentation():
    paper = Result('p', 'Same title', 'https://arxiv.org/abs/1', identifiers={'arxiv': '1'}, kind='paper',
                   code_urls=['https://github.com/org/repo'], tasks=['forecasting'])
    repo = Result('r', 'Same title', 'https://github.com/org/repo', identifiers={'github': '1'}, kind='repository')
    assert len(fuse([paper, repo])) == 2
    results = budget_results([paper, repo], 5, 500)
    assert evidence_tokens(results) <= 500
    from slingsearch.output import markdown
    assert 'Code: https://github.com/org/repo' in markdown(results)


@pytest.mark.parametrize('resource,kind,prefix', [('models', 'model', ''), ('datasets', 'dataset', 'datasets/')])
async def test_huggingface_hub_resources_use_original_cards(resource, kind, prefix):
    def handle(request):
        if request.url.path.startswith('/api/'):
            return httpx.Response(200, json=[{'id': 'org/asset', 'author': 'org'}])
        assert request.url.path == '/' + prefix + 'org/asset/raw/main/README.md'
        return httpx.Response(200, text='# Original source card\nUseful details.')
    async with Client(Config(api_keys={}, huggingface_resource=resource), transport=httpx.MockTransport(handle)) as client:
        response = await client.asearch('diffusion', providers=['huggingface'], ranker='rrf')
    assert response.results[0].kind == kind
    assert 'Original source card' in response.results[0].passages[0]
