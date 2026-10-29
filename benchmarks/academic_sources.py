"""Consolidate SerpAPI, PWC and HF papers, keeping GitHub implementation results separate.

Reuses the original four-provider frozen searches. Only Voyage is used for the
expanded pool; the original Jev/D1 rankings are never reused on changed candidates.
"""
import asyncio
from copy import deepcopy
from dataclasses import asdict
import getpass
import json
from pathlib import Path
from time import perf_counter
import httpx
from benchmarks import academic_eval as base
from benchmarks.datasets import load_snapshot
from benchmarks.decision_budget import Transport
from benchmarks.paperfinding_eval import metadata_id
from slingsearch import Config
from slingsearch.fusion import fuse, normalize
from slingsearch.output import budget_results, evidence_tokens
from slingsearch.passages import select
from slingsearch.providers import PROVIDERS
from slingsearch.rankers import make_ranker

ROOT = Path('benchmarks/reports/academic-sources')
ADDITIONS = ['serpapi', 'papers_with_code', 'huggingface', 'github']


async def collect(client, keys):
    ROOT.mkdir(parents=True, exist_ok=True)
    queries = load_snapshot(base.ROOT / 'retrieval.jsonl')
    done = {q.id for q in load_snapshot(ROOT / 'retrieval.jsonl')} if (ROOT / 'retrieval.jsonl').exists() else set()
    for q in queries:
        if q.id in done: continue
        async def retrieve(name):
            start = perf_counter()
            try:
                rows = await PROVIDERS[name](client, keys.get(name, ''), 'scholar').search(q.question, 10)
                q.retrieval[name] = {'success': True, 'count': len(rows)}
            except Exception as error:
                rows = []
                q.retrieval[name] = {'success': False, 'error_type': type(error).__name__}
            q.retrieval[name]['latency_s'] = perf_counter() - start
            return rows
        raw = [normalize(r) for rows in await asyncio.gather(*(retrieve(n) for n in ADDITIONS)) for r in rows]
        with (ROOT / 'raw-retrieved.jsonl').open('a') as f:
            f.write(json.dumps({'query_id': q.id, 'results': [asdict(r) for r in raw]}) + '\n')
        papers = [r for r in raw if r.kind != 'repository']
        details = await base.metadata(client, keys, sorted({pid for r in papers if (pid := base.lookup_id(r))}))
        accepted = []
        for r in papers:
            paper = details.get(base.lookup_id(r))
            if paper:
                r.identifiers.update(base.aliases(paper))
                r.doi = r.doi or r.identifiers.get('doi')
                r.year = paper.get('year') or r.year
                r.abstract = paper.get('abstract') or r.abstract
                normalize(r)
            if q.id.startswith('litqa2_'):
                date = paper.get('publicationDate') if paper else None
                if not (date < '2024-10-17' if date else r.year is not None and r.year < 2024): continue
            accepted.append(r)
        q.candidates = fuse(q.candidates + accepted)
        q.retrieval['implementation_audit'] = {'github_repositories': sum(r.kind == 'repository' for r in raw),
            'papers_with_code_links': sum(bool(r.code_urls) for r in q.candidates),
            'papers_with_dataset_links': sum(bool(r.dataset_urls) for r in q.candidates)}
        with (ROOT / 'retrieval.jsonl').open('a') as f: f.write(json.dumps(asdict(q)) + '\n')
        print(json.dumps({'query': q.id, 'phase': 'new_sources', 'unique_papers': len(q.candidates),
                          'status': {n: q.retrieval[n] for n in ADDITIONS}}), flush=True)
    (ROOT / 'gold-metadata.json').write_bytes((base.ROOT / 'gold-metadata.json').read_bytes())


async def rank(client, key):
    gold = json.loads((ROOT / 'gold-metadata.json').read_text())
    path = ROOT / 'rankings.jsonl'
    records = {(r['query_id'], r['method']): r for r in [json.loads(s) for s in path.read_text().splitlines()]} if path.exists() else {}
    providers = base.PROVIDERS_USED + ADDITIONS[:-1]
    for q in load_snapshot(ROOT / 'retrieval.jsonl'):
        pool = deepcopy(q.candidates[:50])
        for r in pool: r.passages = select(q.question, [r.abstract] if r.abstract else r.passages)
        singles = {p: sorted([r for r in q.candidates if any(s.provider == p for s in r.sources)],
                      key=lambda r: min(s.rank for s in r.sources if s.provider == p)) for p in providers}
        for docs in singles.values():
            for r in docs: r.passages = select(q.question, [r.abstract] if r.abstract else r.passages)
        runs = {'rrf': pool, **{'single:' + p: rows for p, rows in singles.items()}}
        for method in [*runs, 'voyage', *[p + '+voyage' for p in providers]]:
            if records.get((q.id, method), {}).get('success'): continue
            backend = None
            try:
                if method in runs:
                    ranked = runs[method]
                    if method.startswith('single:') and not q.retrieval[method.split(':')[1]]['success']:
                        raise ValueError('Retrieval failed')
                else:
                    docs = singles[method.split('+')[0]] if '+' in method else pool
                    if '+' in method and not q.retrieval[method.split('+')[0]]['success']:
                        raise ValueError('Retrieval failed')
                    backend = make_ranker('voyage', client, Config(api_keys={'openrouter': key}, candidate_tokens=400))
                    ranked = await backend.rank(q.question, docs) if docs else []
                evidence = budget_results(ranked, 5, 1500)
                row = {'query_id': q.id, 'method': method, 'success': True,
                    'result_ids': [r.id for r in ranked], 'metrics': base.score(q, ranked, gold),
                    'product_metrics': base.score(q, evidence, gold), 'output_tokens': evidence_tokens(evidence)}
            except Exception as error:
                row = {'query_id': q.id, 'method': method, 'success': False, 'error_type': type(error).__name__}
            row.update(reported_cost_usd=backend.cost_usd if backend else 0, usage=backend.usage if backend else {})
            records[q.id, method] = row
            with path.open('a') as f: f.write(json.dumps(row) + '\n')
            await asyncio.sleep(.4)
        print(json.dumps({'query': q.id, 'phase': 'expanded_ranking'}), flush=True)


async def run(key):
    keys = Config().api_keys
    async with httpx.AsyncClient(transport=Transport(base.BUDGET), timeout=60) as client:
        await collect(client, keys)
        await rank(client, key)


if __name__ == '__main__':
    asyncio.run(run(getpass.getpass('OpenRouter key: ')))
