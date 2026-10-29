"""Bounded shortlist enrichment: evidence additions never count as retrieval votes."""
from urllib.parse import urlsplit
from .providers.semantic_scholar import SemanticScholar
from .fusion import normalize


def paper_identifier(result):
    ids = result.identifiers
    if ids.get('semantic_scholar'): return ids['semantic_scholar']
    if ids.get('corpusid'): return 'CorpusId:'+ids['corpusid']
    if result.doi: return 'DOI:'+result.doi
    if ids.get('arxiv'): return 'ARXIV:'+ids['arxiv']
    if urlsplit(result.url).hostname in ('arxiv.org','www.arxiv.org','www.semanticscholar.org','semanticscholar.org','www.aclweb.org','www.acm.org','www.biorxiv.org'):
        return 'URL:'+result.url
    return None


async def enrich(client, config, question, candidates, bounded, year=None):
    eligible = [r for r in candidates if paper_identifier(r)]
    status = {'status':'complete','requested_papers':len(eligible),'enriched_papers':0,
              'stage':'before_global_ranking','warnings':[]}
    if not eligible:return status
    adapter = SemanticScholar(client, config.api_keys.get('semantic_scholar',''), 'scholar',year)
    try:
        rows = await bounded(adapter.search_snippets(question, min(100,len(eligible)*3),
                            paper_ids=list(dict.fromkeys(paper_identifier(r) for r in eligible))))
        status['retrieval_version'] = adapter.snippet_version
        status['warnings'] = adapter.snippet_warnings
        for candidate in eligible:
            added = False
            for row in rows:
                row = normalize(row)
                # Matching restricted snippets still requires verified identities;
                # title coincidence alone must not attach evidence to a paper.
                match = (candidate.doi and row.doi == candidate.doi or
                         any(row.identifiers.get(k)==v for k,v in candidate.identifiers.items()))
                if not match:continue
                for index,text in enumerate(row.passages):
                    if text not in candidate.passages:candidate.passages.append(text)
                    target=candidate.passages.index(text)
                    for entry in row.passage_metadata:
                        if entry['index']==index:
                            mapped={**entry,'index':target}
                            if mapped not in candidate.passage_metadata:candidate.passage_metadata.append(mapped)
                    added=True
            status['enriched_papers'] += int(added)
    except Exception as error:
        status.update(status='unavailable',error_type=type(error).__name__)
    return status
