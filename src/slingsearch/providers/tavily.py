from .base import Provider


class Tavily(Provider):
    name = 'tavily'
    async def search(self, question, limit=10):
        data = await self.request('POST', 'https://api.tavily.com/search',
                                  headers={'Authorization': f'Bearer {self.key}'},
                                  json={'query': question, 'max_results': min(limit, 20), 'include_answer': False})
        return [self.result({'title': r.get('title') or '', 'url': r['url'],
                             'passages': [r.get('content') or '']}, i)
                for i, r in enumerate(data['results'][:limit], 1)]
