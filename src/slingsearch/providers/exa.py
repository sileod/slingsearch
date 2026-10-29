from .base import Provider


class Exa(Provider):
    name = 'exa'
    async def search(self, question, limit=10):
        body = {'query': question, 'numResults': limit, 'type': 'auto',
                'contents': {'text': {'maxCharacters': 6000}, 'highlights': True}}
        if self.mode == 'scholar':
            body['category'] = 'research paper'
        if self.year:
            body.update(startPublishedDate=f'{self.year[0]}-01-01T00:00:00Z',
                        endPublishedDate=f'{self.year[1]}-12-31T23:59:59Z')
        data = await self.request('POST', 'https://api.exa.ai/search', headers={'x-api-key': self.key}, json=body)
        self.cost_usd = (data.get('costDollars') or {}).get('total')
        return [self.result({'title': r.get('title') or '', 'url': r['url'],
                             'passages': r.get('highlights') or [r.get('text') or ''],
                             'year': int(r['publishedDate'][:4]) if r.get('publishedDate') else None,
                             'authors': [r['author']] if r.get('author') else []}, i)
                for i, r in enumerate(data['results'][:limit], 1)]
