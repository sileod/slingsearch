from .base import Provider


class Parallel(Provider):
    name = 'parallel'
    async def search(self, question, limit=10):
        objective = ('Find scholarly research papers addressing this question, prioritizing '
                     'primary papers and abstracts: ' + question) if self.mode == 'scholar' else question
        data = await self.request('POST', 'https://api.parallel.ai/v1/search', headers={'x-api-key': self.key},
                                  json={'objective': objective, 'search_queries': [question], 'mode': 'basic',
                                        'advanced_settings': {'max_results': limit,
                                                              'excerpt_settings': {'max_chars_per_result': 6000}}})
        return [self.result({'title': r.get('title') or '', 'url': r['url'],
                             'passages': r.get('excerpts') or []}, i)
                for i, r in enumerate(data['results'][:limit], 1)]
