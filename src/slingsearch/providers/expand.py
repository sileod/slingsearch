"""Expand web search; provider-native scores are not global relevance scores."""
from .base import Provider


class Expand(Provider):
    name = 'expand'

    async def search(self, question, limit=10):
        if not isinstance(question, str) or not question.strip() or len(question.strip()) > 2000:
            raise ValueError('Expand requires a nonempty query of at most 2000 characters')
        if type(limit) is not int or limit < 1:
            raise ValueError('limit must be a positive integer')
        # The API supports at most ten results and has no pagination. Do not
        # repeat calls to simulate independent results or silently rewrite queries.
        data = await self.request('POST', 'https://api.expand.ai/v1/search',
            headers={'x-expand-api-key':self.key},
            json={'query':question.strip(),'maxResults':min(limit,10)})
        results = []
        for rank, row in enumerate(data.get('results') or [],1):
            if not row.get('url'):continue
            excerpt = row.get('excerpt') or ''
            results.append(self.result({'title':row.get('title') or '', 'url':row['url'],
                'passages':[excerpt] if excerpt else [],
                'passage_metadata':[{'index':0,'provider':self.name,'snippet_kind':'search_excerpt'}] if excerpt else []},rank))
            if len(results) >= min(limit,10):break
        return results
