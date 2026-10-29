from .base import Provider


def abstract_from_index(index):
    if not index:
        return None
    positions = {p: word for word, locations in index.items() for p in locations}
    return ' '.join(positions[p] for p in sorted(positions))


class OpenAlex(Provider):
    name = 'openalex'
    rate_limit_delay = 2.0

    def __init__(self, client, key='', mode='scholar', year=None, *, semantic=True):
        super().__init__(client, key, mode, year)
        self.semantic = semantic

    async def search(self, question, limit=10):
        if self.semantic and limit > 50:
            raise ValueError('OpenAlex semantic search supports at most 50 results')
        search_parameter = 'search.semantic' if self.semantic else 'search'
        params = {search_parameter: question, 'per_page': min(limit, 50 if self.semantic else 100),
                  'select': 'id,doi,title,authorships,publication_year,primary_location,abstract_inverted_index,cited_by_count,ids'}
        if self.year:
            params['filter'] = f'publication_year:{self.year[0]}-{self.year[1]}'
        headers = {'Authorization': f'Bearer {self.key}'} if self.key else {}
        results = []
        for page in range(1, 2 if self.semantic or not self.key else 11):
            params['page'] = page
            data = await self.request('GET', 'https://api.openalex.org/works', params=params, headers=headers)
            cost = data.get('meta', {}).get('cost_usd')
            if cost is not None:
                self.cost_usd = (self.cost_usd or 0) + cost
            for r in data['results']:
                location = r.get('primary_location') or {}
                abstract = abstract_from_index(r.get('abstract_inverted_index'))
                results.append(self.result({'title': r.get('title') or '',
                    'url': r.get('doi') or location.get('landing_page_url') or r['id'], 'doi': r.get('doi'),
                    'abstract': abstract, 'authors': [a['author']['display_name'] for a in r.get('authorships', [])],
                    'year': r.get('publication_year'), 'venue': (location.get('source') or {}).get('display_name'),
                    'citation_count': r.get('cited_by_count'), 'identifiers': r.get('ids') or {'openalex': r['id']}}, len(results) + 1))
            if len(results) >= limit or len(data['results']) < params['per_page']:
                break
        return results[:limit]
