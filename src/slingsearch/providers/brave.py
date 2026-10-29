from .base import Provider


class Brave(Provider):
    name = 'brave'
    async def search(self, question, limit=10):
        results = []
        for offset in range(10):
            data = await self.request('GET', 'https://api.search.brave.com/res/v1/web/search',
                                      headers={'X-Subscription-Token': self.key},
                                      params={'q': question, 'count': min(limit, 20), 'offset': offset,
                                              'extra_snippets': 'true', 'text_decorations': 'false'})
            rows = data.get('web', {}).get('results', [])
            for r in rows:
                results.append(self.result({'title': r.get('title') or '', 'url': r['url'],
                    'passages': [r.get('description') or '', *r.get('extra_snippets', [])]}, len(results) + 1))
            if len(results) >= limit or not rows or not data.get('query', {}).get('more_results_available'):
                break
        return results[:limit]
