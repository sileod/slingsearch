"""Python port of Search CLI's URL-only RRF for controlled frozen replay.

Algorithm adapted from paperfoot/search-cli src/engine.rs (MIT).
This is NOT execution of the Rust CLI or an end-to-end latency comparison.
"""
from copy import deepcopy

TRACKERS = {'fbclid','gclid','gclsrc','dclid','msclkid','igshid','mc_cid','mc_eid',
            'ref_src','ref_url','cmpid','_hsenc','_hsmi'}


def normalize_url(url):
    value = url.rstrip('/').lower()
    for prefix in ('https://', 'http://'):
        if value.startswith(prefix):
            value = value[len(prefix):]
            break
    value = value.removeprefix('www.')
    if '?' not in value: return value
    base, query = value.split('?', 1)
    parts = [p for p in query.split('&') if not p.split('=')[0].startswith('utm_') and p.split('=')[0] not in TRACKERS]
    return base + ('?' + '&'.join(parts) if parts else '')


def rank(buckets):
    grouped = {}
    # Upstream breaks score ties by bucket/item insertion order.
    for provider, items in buckets:
        for index, item in enumerate(items, 1):
            key = normalize_url(item.url)
            if key not in grouped:
                grouped[key] = [deepcopy(item), 0., len(grouped)]
            grouped[key][1] += 1 / (60 + index)
    return [row[0] for row in sorted(grouped.values(), key=lambda row: (-row[1], row[2]))]


def replay(root):
    """Use the same enriched identities/date acceptance as all competing methods."""
    import json
    from pathlib import Path
    from benchmarks.datasets import load_snapshot
    from benchmarks.academic_eval import score
    from slingsearch import Result
    from slingsearch.output import budget_results, evidence_tokens
    root = Path(root)
    gold = json.loads((root / 'gold-metadata.json').read_text())
    raw_files = [root / 'raw-retrieved.jsonl']
    if root.name == 'academic-sources':
        raw_files.insert(0, root.parent / 'academic-samples/raw-retrieved.jsonl')
    raw_by_file = []
    for path in raw_files:
        raw_by_file.append({r['query_id']: r['results'] for r in [json.loads(l) for l in path.read_text().splitlines()]})
    for query in load_snapshot(root / 'retrieval.jsonl'):
        by_slot = {(s.provider, s.rank): result for result in query.candidates for s in result.sources}
        buckets = {}
        for lookup in raw_by_file:
            for row in lookup.get(query.id, []):
                result = Result.from_dict(row)
                for source in result.sources:
                    enriched = by_slot.get((source.provider, source.rank))
                    if enriched is None or enriched.kind == 'repository': continue
                    # Identity enrichment is held constant; source URL controls this baseline's fusion.
                    result.identifiers = enriched.identifiers.copy()
                    result.doi = enriched.doi
                    result.abstract = enriched.abstract
                    result.passages = enriched.passages.copy()
                    result.id = enriched.id
                    buckets.setdefault(source.provider, []).append((source.rank, result))
        ordered = rank([(p, [r for _, r in sorted(rows, key=lambda item: item[0])]) for p, rows in sorted(buckets.items())])[:50]
        evidence = budget_results(ordered, 5, 1500)
        row = {'query_id': query.id, 'method': 'search_cli_fusion_port', 'success': True,
               'metrics': score(query, ordered, gold), 'product_metrics': score(query, evidence, gold),
               'result_ids': [r.id for r in ordered], 'output_tokens': evidence_tokens(evidence),
               'reported_cost_usd': 0, 'latency_s': None,
               'scope': 'Python port of URL-only fusion; shared frozen providers and metadata, not the Rust CLI'}
        with (root / 'rankings.jsonl').open('a') as f: f.write(json.dumps(row) + '\n')


if __name__ == '__main__':
    import sys
    replay(sys.argv[1])
