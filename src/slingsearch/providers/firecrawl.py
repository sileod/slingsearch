from .base import Provider


class Firecrawl(Provider):
    name = 'firecrawl'
    async def search(self, question, limit=10):
        data = await self.request('POST', 'https://api.firecrawl.dev/v2/search',
                                  headers={'Authorization': f'Bearer {self.key}'},
                                  json={'query': question, 'limit': limit, 'sources': ['web']})
        if data.get('success') is False:
            raise ValueError('Search rejected')
        rows = data['data']['web']
        return [self.result({'title': r.get('title') or '', 'url': r['url'],
                             'passages': [r.get('markdown') or r.get('description') or '']}, i)
                for i, r in enumerate(rows[:limit], 1)]
