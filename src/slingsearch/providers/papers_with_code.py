"""Explicit Hugging Face-maintained Papers with Code catalog backend (.co).

The retired .com API is not used. Detail reads supply abstracts and linked code;
search results alone do not contain these fields. No TLDRs enter source passages.
"""
from urllib.parse import quote
from .base import Provider
from .queries import scholar_keywords

BASE = 'https://paperswithcode.co/api/v1'


class PapersWithCode(Provider):
    name = 'papers_with_code'
    rate_limit_delay = 2.0

    async def search(self, question, limit=10):
        params = {'q': scholar_keywords(question, max_terms=4)[:300], 'mode': 'keyword', 'page_size': min(limit, 100), 'page': 1}
        if self.year:
            params.update(start_date=f'{self.year[0]}-01-01', end_date=f'{self.year[1]}-12-31')
        results = []
        while len(results) < limit:
            data = await self.request('GET', BASE + '/papers/search', params=params)
            if not data.get('results') and not results and params['q'] != scholar_keywords(question, max_terms=2)[:300]:
                params['q'] = scholar_keywords(question, max_terms=2)[:300]
                continue
            for summary in data.get('results', []):
                identifier = summary.get('arxiv_id') or summary['id']
                row = await self.request('GET', BASE + '/papers/' + quote(str(identifier), safe=''),
                                         params={'include_resources': 'true'})
                published = row.get('published')
                repositories = row.get('repositories') or []
                code = [repo['url'] for repo in repositories if repo.get('url')]
                ids = {'papers_with_code': str(row['id'])}
                if row.get('arxiv_id'): ids['arxiv'] = row['arxiv_id']
                if row.get('semantic_scholar_id'): ids['semantic_scholar'] = row['semantic_scholar_id']
                results.append(self.result({'title': row['title'], 'kind': 'paper',
                    'url': row.get('url_abs') or row.get('source_url') or 'https://paperswithcode.co/paper/' + str(identifier),
                    'abstract': row.get('abstract'), 'authors': row.get('authors') or [],
                    'year': int(published[:4]) if published else None, 'identifiers': ids,
                    'venue': row.get('conference_name') or row.get('proceeding'),
                    'citation_count': row.get('citation_count'), 'code_urls': code,
                    'tasks': [task['name'] for task in row.get('tasks') or [] if task.get('name')],
                    'dataset_urls': row.get('hf_datasets') or [],
                    'related_urls': [r['url'] for r in row.get('project_pages') or [] if r.get('url')],
                }, len(results) + 1))
                if len(results) >= limit: break
            if not data.get('next_page') or not data.get('results'): break
            params['page'] = data['next_page']
        return results
