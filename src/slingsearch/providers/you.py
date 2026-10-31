"""You.com web search with query-aware highlights, not generated answers."""
from .base import Provider


class You(Provider):
    name = 'you'

    async def search(self, question, limit=10):
        if not isinstance(question,str) or not question.strip():
            raise ValueError('question must be nonempty')
        if type(limit) is not int or not 1 <= limit <= 50:
            raise ValueError('You highlights supports limits from 1 to 50')
        # Highlights are included in base search pricing. Do not request live
        # full-page extraction, answer generation, or the Research API.
        data = await self.request('POST','https://ydc-index.io/v1/search',
            headers={'X-API-Key':self.key},json={'query':question,'count':limit,
                'extraction':{'extraction_mode':'highlights'}})
        results=[]
        for rank,row in enumerate((data.get('results') or {}).get('web') or [],1):
            if not row.get('url'):continue
            contents=row.get('contents') or {}
            texts=contents.get('highlights') or row.get('snippets') or []
            if isinstance(texts,str):texts=[texts]
            texts=[text for text in texts if isinstance(text,str) and text.strip()]
            if not texts and row.get('description'):texts=[row['description']]
            # page_age is page metadata, not a reliable publication year for a
            # paper. Never use it to pass a scholarly year filter.
            results.append(self.result({'title':row.get('title') or '', 'url':row['url'],
                'passages':texts, 'passage_metadata':[{'index':i,'provider':self.name,
                    'snippet_kind':'highlight' if contents.get('highlights') else 'search_excerpt'} for i in range(len(texts))]},rank))
            if len(results)>=limit:break
        return results
