"""Hugging Face paper discovery; summaries are source abstracts, not AI summaries."""
from .base import Provider
from .queries import scholar_keywords


class HuggingFace(Provider):
    name = 'huggingface'
    rate_limit_delay = 2.0

    def __init__(self, *args, resource='papers', **kwargs):
        super().__init__(*args, **kwargs)
        if resource not in ('papers', 'models', 'datasets'):
            raise ValueError('Unknown Hugging Face resource')
        self.resource = resource

    async def search(self, question, limit=10):
        if self.resource != 'papers':
            return await self.search_hub(question, limit)
        headers = {'Authorization': f'Bearer {self.key}'} if self.key else {}
        data = await self.request('GET', 'https://huggingface.co/api/papers/search',
            params={'q': question[:250], 'limit': min(limit, 120)}, headers=headers)
        results = []
        for item in data:
            row = item.get('paper') or item
            published = row.get('publishedAt') or ''
            year = int(published[:4]) if published else None
            if self.year and (year is None or not self.year[0] <= year <= self.year[1]): continue
            arxiv = row['id']
            results.append(self.result({'title': row['title'], 'url': 'https://arxiv.org/abs/' + arxiv,
                'abstract': row.get('summary'), 'authors': [a['name'] for a in row.get('authors', [])],
                'year': year, 'identifiers': {'arxiv': arxiv}, 'kind': 'paper',
                'code_urls': [row['githubRepo']] if row.get('githubRepo') else [],
                'related_urls': ['https://huggingface.co/papers/' + arxiv]}, len(results) + 1))
        return results[:limit]

    async def search_hub(self, question, limit):
        headers = {'Authorization': f'Bearer {self.key}'} if self.key else {}
        rows = await self.request('GET', 'https://huggingface.co/api/' + self.resource,
            params={'search': scholar_keywords(question, max_terms=3), 'limit': limit}, headers=headers)
        results = []
        for row in rows:
            rid = row.get('id') or row.get('modelId')
            prefix = 'datasets/' if self.resource == 'datasets' else ''
            url = 'https://huggingface.co/' + prefix + rid
            response = await self.client.get(url + '/raw/main/README.md', headers=headers, follow_redirects=True)
            passages = [response.text[:12000]] if response.is_success else []
            results.append(self.result({'title': rid, 'url': url, 'passages': passages,
                'kind': 'dataset' if self.resource == 'datasets' else 'model',
                'identifiers': {'huggingface_' + self.resource: rid},
                'authors': [row['author']] if row.get('author') else []}, len(results) + 1))
        return results
