"""Public repository discovery; never claims to search repository code anonymously."""
from .base import Provider
from .queries import scholar_keywords


class GitHub(Provider):
    name = 'github'

    async def search(self, question, limit=10):
        headers = {'Accept': 'application/vnd.github+json', 'X-GitHub-Api-Version': '2022-11-28'}
        if self.key: headers['Authorization'] = f'Bearer {self.key}'
        query = scholar_keywords(question, max_terms=4)
        params = {'q': query, 'per_page': min(limit, 100), 'page': 1}
        results = []
        while len(results) < limit:
            data = await self.request('GET', 'https://api.github.com/search/repositories', params=params, headers=headers)
            if not data.get('items') and not results and params['q'] != scholar_keywords(question, max_terms=2):
                params['q'] = scholar_keywords(question, max_terms=2)
                continue
            for row in data.get('items', []):
                results.append(self.result({'title': row.get('full_name') or row['name'], 'url': row['html_url'],
                    'passages': [row['description']] if row.get('description') else [],
                    'authors': [row['owner']['login']] if row.get('owner') else [],
                    'identifiers': {'github': str(row['id'])}, 'code_urls': [row['html_url']], 'kind': 'repository',
                }, len(results) + 1))
            if len(results) >= limit or len(data.get('items', [])) < params['per_page']:
                break
            params['page'] += 1
        return results[:limit]
