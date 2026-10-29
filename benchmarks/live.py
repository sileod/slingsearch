"""Collect normalized provider outputs once, before comparing ranking methods."""
import argparse
import asyncio
from dataclasses import asdict
import json
from pathlib import Path
from time import perf_counter
import httpx
from slingsearch.config import Config
from slingsearch.providers import PROVIDERS
from .datasets import load_snapshot
from .datasets.litsearch import load as load_litsearch
from .evaluate import match_paper
from slingsearch.fusion import normalize


async def snapshot(queries, providers, *, config=None, limit=10, corpus=None, transport=None):
    config = config or Config()
    if any(p not in PROVIDERS for p in providers):
        raise ValueError('Unknown provider')
    async with httpx.AsyncClient(timeout=config.timeout, transport=transport) as client:
        for q in queries:
            q.candidates, q.retrieval = [], {}
            async def retrieve(name):
                key = config.api_keys.get(name, '')
                if not key and name not in ('openalex', 'semantic_scholar'):
                    q.retrieval[name] = {'success': False, 'skipped': True, 'reason': 'missing credentials'}
                    return []
                provider = PROVIDERS[name](client, key, q.mode)
                start = perf_counter()
                try:
                    results = await provider.search(q.question, limit)
                    for result in results:
                        normalize(result)
                        if corpus:
                            match = match_paper(result, corpus)
                            if match:
                                result.identifiers['corpusid'] = match.removeprefix('corpusid:')
                    q.retrieval[name] = {'success': True, 'latency': perf_counter() - start, 'cost': provider.cost_usd}
                    return results
                except Exception as error:
                    q.retrieval[name] = {'success': False, 'latency': perf_counter() - start,
                                         'error': type(error).__name__, 'http_status': getattr(getattr(error, 'response', None), 'status_code', None),
                                         'cost': provider.cost_usd}
                    return []
            batches = await asyncio.gather(*(retrieve(name) for name in dict.fromkeys(providers)))
            q.candidates = [r for batch in batches for r in batch]
    return queries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('queries', help='Normalized query JSONL, or native LitSearch queries with --litsearch')
    parser.add_argument('--providers', nargs='+', required=True, choices=list(PROVIDERS))
    parser.add_argument('--output', required=True)
    parser.add_argument('--litsearch', action='store_true')
    parser.add_argument('--corpus', help='JSON map of corpusid:ID to paper metadata (doi, title, year, identifiers)')
    parser.add_argument('--limit', type=int, default=10)
    args = parser.parse_args()
    queries = load_litsearch(args.queries) if args.litsearch else load_snapshot(args.queries)
    corpus = json.loads(Path(args.corpus).read_text()) if args.corpus else None
    queries = asyncio.run(snapshot(queries, args.providers, limit=args.limit, corpus=corpus))
    Path(args.output).write_text(''.join(json.dumps(asdict(q)) + '\n' for q in queries))


if __name__ == '__main__':
    main()
