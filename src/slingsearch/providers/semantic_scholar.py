import asyncio
import httpx
from time import monotonic
from .base import Provider
from .queries import scholar_keywords


class SemanticScholar(Provider):
    name = 'semantic_scholar'
    rate_limit_delay = 2.0
    async def request(self, method, url, **kwargs):
        # S2's key quota is shared by search, snippets, metadata and graphs.
        # Serialize and pace these operations on a reused HTTP client.
        if not hasattr(self.client, '_s2_request_lock'):
            self.client._s2_request_lock = asyncio.Lock()
            self.client._s2_last_request = 0.0
        async with self.client._s2_request_lock:
            delay = 1.05 - (monotonic() - self.client._s2_last_request)
            if delay > 0: await asyncio.sleep(delay)
            try:
                return await super().request(method, url, **kwargs)
            finally:
                self.client._s2_last_request = monotonic()

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

    async def search_snippets(self, question, limit=10, *, paper_ids=None):
        """A limit counts snippets, not unique papers; this endpoint has no offset."""
        if type(limit) is not int or not 1 <= limit <= 1000:
            raise ValueError('Snippet limit must be from 1 to 1000')
        if paper_ids is not None and (not isinstance(paper_ids, (list, tuple)) or not paper_ids or len(paper_ids) > 100 or
                any(not isinstance(pid, str) or not pid or ',' in pid for pid in paper_ids)):
            raise ValueError('Use one to 100 comma-free paper identifiers')
        params = {'query': question, 'limit': limit,
                  'fields':'snippet.text,snippet.snippetKind,snippet.section,snippet.snippetOffset'}
        if paper_ids is not None: params['paperIds'] = ','.join(paper_ids)
        if self.year: params['year'] = f'{self.year[0]}-{self.year[1]}'
        headers = {'x-api-key':self.key} if self.key else {}
        self.snippet_warnings = []
        endpoint = 'https://api.semanticscholar.org/graph/v1/snippet/search'
        try:
            data = await self.request('GET',endpoint,params=params,headers=headers)
        except httpx.HTTPStatusError as error:
            # Despite the plain-text contract, literal mathematical asterisks
            # can trigger unsupported wildcard syntax. Retry only this case.
            if error.response.status_code != 400 or '*' not in question: raise
            params['query'] = question.replace('*',' ')
            data = await self.request('GET',endpoint,params=params,headers=headers)
            self.snippet_warnings.append('Snippet query retried without literal asterisks after HTTP 400; ranking retains the original question')
        self.snippet_version = data.get('retrievalVersion')
        rows = []
        for rank, item in enumerate(data.get('data') or [],1):
            paper, snippet = item.get('paper') or {}, item.get('snippet') or {}
            if not paper.get('corpusId') or not snippet.get('text'): continue
            corpus = str(paper['corpusId'])
            rows.append(self.result({'title':paper.get('title') or '',
                'url':'https://www.semanticscholar.org/paper/CorpusID:'+corpus,
                'identifiers':{'corpusid':corpus},
                'authors':[a if isinstance(a,str) else a['name'] for a in paper.get('authors') or [] if isinstance(a,str) or a.get('name')],
                'passages':[snippet['text']], 'passage_metadata':[{
                    'index':0, 'provider':self.name, 'section':snippet.get('section'),
                    'snippet_kind':snippet.get('snippetKind'), 'offset':snippet.get('snippetOffset'),
                    'retrieval_version':self.snippet_version, 'license':(paper.get('openAccessInfo') or {}).get('license')}]},rank))
        # Snippets omit DOI/year/abstract. One bounded metadata batch gives stable
        # identifiers for merging and matching restricted enrichment results.
        ids = list(dict.fromkeys('CorpusId:'+r.identifiers['corpusid'] for r in rows))
        if ids:
            try:
                papers = []
                for start in range(0,len(ids),500):
                    papers.extend(await self.request('POST','https://api.semanticscholar.org/graph/v1/paper/batch',
                        params={'fields':'title,url,abstract,authors,year,venue,citationCount,externalIds,corpusId'},
                        json={'ids':ids[start:start+500]},headers=headers))
                lookup = {pid:paper for pid,paper in zip(ids,papers) if paper}
                for result in rows:
                    paper = lookup.get('CorpusId:'+result.identifiers['corpusid'])
                    if not paper: continue
                    result.identifiers.update({k.lower():str(v) for k,v in (paper.get('externalIds') or {}).items() if v})
                    if paper.get('paperId'):result.identifiers['semantic_scholar']=paper['paperId']
                    result.doi=result.identifiers.get('doi')
                    result.url=paper.get('url') or result.url
                    result.abstract=paper.get('abstract')
                    result.year=paper.get('year');result.venue=paper.get('venue');result.citation_count=paper.get('citationCount')
            except Exception as error:
                self.snippet_warnings.append('Snippet metadata lookup failed ('+type(error).__name__+'); corpus identifiers retained')
        return rows
