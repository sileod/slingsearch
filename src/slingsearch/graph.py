"""Explicit bounded one-hop citation graph, with source-backed paper metadata."""
from dataclasses import asdict
from urllib.parse import quote
from .fusion import fuse, normalize
from .providers.semantic_scholar import SemanticScholar

FIELDS = 'title,url,abstract,authors,year,venue,citationCount,externalIds'


def paper_result(adapter, row, rank):
    ids = {k.lower(): str(v) for k, v in (row.get('externalIds') or {}).items() if v}
    ids['semantic_scholar'] = row['paperId']
    return normalize(adapter.result({'title': row.get('title') or '',
        'url': row.get('url') or 'https://www.semanticscholar.org/paper/' + row['paperId'],
        'abstract': row.get('abstract'), 'doi': ids.get('doi'),
        'authors': [a['name'] for a in row.get('authors', []) if a.get('name')],
        'year': row.get('year'), 'venue': row.get('venue'),
        'citation_count': row.get('citationCount'), 'identifiers': ids}, rank))


async def citation_graph(client, config, *, paper_id, direction='both', limit=10):
    if not isinstance(paper_id, str) or not paper_id.strip() or len(paper_id) > 2048:
        raise ValueError('paper_id must be a nonempty paper identifier')
    if direction not in ('references', 'citations', 'both') or type(limit) is not int or not 1 <= limit <= 100:
        raise ValueError('Use references, citations or both, and a limit from 1 to 100 per direction')
    adapter = SemanticScholar(client, config.api_keys.get('semantic_scholar', ''), 'scholar')
    headers = {'x-api-key': adapter.key} if adapter.key else {}
    base = 'https://api.semanticscholar.org/graph/v1/paper/'
    result = {'seed': paper_id, 'nodes': [], 'edges': [], 'successful': [], 'failed': {},
              'warnings': [], 'limit_per_direction': limit, 'truncated': {}}
    try:
        row = await adapter.request('GET', base + quote(paper_id, safe=''), headers=headers, params={'fields': FIELDS})
        seed = paper_result(adapter, row, 1)
    except Exception as error:
        result['failed']['seed'] = type(error).__name__
        return result
    result['seed'] = seed.id
    nodes, edges = [seed], set()
    for kind in (['references', 'citations'] if direction == 'both' else [direction]):
        try:
            data = await adapter.request('GET', base + quote(row['paperId'], safe='') + '/' + kind,
                headers=headers, params={'fields': FIELDS, 'limit': limit, 'offset': 0})
            key = 'citedPaper' if kind == 'references' else 'citingPaper'
            for rank, item in enumerate(data.get('data', []), 1):
                paper = item.get(key)
                if not paper or not paper.get('paperId'): continue
                node = paper_result(adapter, paper, rank)
                nodes.append(node)
                source, target = (seed.id, node.id) if kind == 'references' else (node.id, seed.id)
                if source != target: edges.add((source, target))
            result['successful'].append(kind)
            result['truncated'][kind] = data.get('next') is not None or len(data.get('data', [])) >= limit
        except Exception as error:
            result['failed'][kind] = type(error).__name__
    result['nodes'] = [asdict(node) for node in fuse(nodes)]
    result['edges'] = [{'source': source, 'target': target, 'relation': 'cites',
                        'provider': 'semantic_scholar'} for source, target in sorted(edges)]
    result['warnings'].append('One-hop graph, bounded per direction; citation links are not relevance judgments or ranking boosts')
    return result
