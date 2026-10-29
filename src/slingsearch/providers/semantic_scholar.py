from .base import Provider
from .queries import scholar_keywords


class SemanticScholar(Provider):
    name = 'semantic_scholar'
    rate_limit_delay = 2.0
    async def search(self, question, limit=10):
        params = {'query': scholar_keywords(question), 'limit': min(limit, 100), 'offset': 0,
                  'fields': 'title,url,abstract,authors,year,venue,citationCount,externalIds'}
        if self.year:
            params['year'] = f'{self.year[0]}-{self.year[1]}'
        headers = {'x-api-key': self.key} if self.key else {}
        results = []
        while len(results) < limit:
            data = await self.request('GET', 'https://api.semanticscholar.org/graph/v1/paper/search', params=params, headers=headers)
            for r in data.get('data', []):
                ids = {k.lower(): str(v) for k, v in (r.get('externalIds') or {}).items() if v}
                ids['semantic_scholar'] = r['paperId']
                results.append(self.result({'title': r.get('title') or '',
                    'url': r.get('url') or 'https://www.semanticscholar.org/paper/' + r['paperId'],
                    'doi': ids.get('doi'), 'abstract': r.get('abstract'), 'identifiers': ids,
                    'authors': [a['name'] for a in r.get('authors', [])], 'year': r.get('year'),
                    'venue': r.get('venue'), 'citation_count': r.get('citationCount')}, len(results) + 1))
            if not data.get('data') or data.get('next') is None:
                break
            params['offset'] = data['next']
        return results[:limit]
