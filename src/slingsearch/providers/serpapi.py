"""Google web and Google Scholar retrieval through SerpAPI."""
import re
from .base import Provider
from .queries import scholar_keywords


class SerpAPI(Provider):
    name = 'serpapi'

    async def search(self, question, limit=10):
        scholar = self.mode == 'scholar'
        page_size = min(limit, 20 if scholar else 10)
        params = {'engine': 'google_scholar' if scholar else 'google', 'q': scholar_keywords(question, max_terms=6) if scholar else question,
                  'api_key': self.key, 'num': page_size, 'start': 0}
        if self.year:
            if scholar:
                params.update(as_ylo=self.year[0], as_yhi=self.year[1])
            else:
                params['tbs'] = f'cdr:1,cd_min:1/1/{self.year[0]},cd_max:12/31/{self.year[1]}'
        results = []
        while len(results) < limit:
            data = await self.request('GET', 'https://serpapi.com/search.json', params=params)
            if data.get('error') == "Google hasn't returned any results for this query.":
                break
            if data.get('error'):
                raise ValueError('SerpAPI returned an API error')
            rows = data.get('organic_results', [])
            for row in rows:
                resources = [r['link'] for r in row.get('resources', []) if r.get('link')]
                url = row.get('link') or next(iter(resources), None)
                if not url:
                    continue  # A citation-only record has no inspectable source URL.
                publication = row.get('publication_info') or {}
                summary = publication.get('summary') or ''
                years = re.findall(r'\b(?:19|20)\d{2}\b', summary)
                inline = row.get('inline_links') or {}
                cited = inline.get('cited_by') or {}
                identifiers = {}
                if cited.get('cites_id'):
                    identifiers['google_scholar_cluster'] = str(cited['cites_id'])
                results.append(self.result({
                    'title': row.get('title') or '', 'url': url,
                    'passages': [row['snippet']] if row.get('snippet') else [],
                    'authors': [a['name'] for a in publication.get('authors', []) if a.get('name')],
                    'year': int(years[-1]) if years else None,
                    'citation_count': cited.get('total'), 'identifiers': identifiers,
                    'related_urls': [link for link in resources if link != url],
                }, len(results) + 1))
            if len(results) >= limit or not rows or not (data.get('serpapi_pagination') or {}).get('next'):
                break
            # Construct pagination ourselves; never follow a URL containing a key.
            params['start'] += page_size
        return results[:limit]
