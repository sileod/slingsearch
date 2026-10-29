"""Experimental Exa-preserving fusion, kept outside the product defaults."""
from slingsearch.fusion import fuse


def anchor(candidates, *, keep=1, weight=1.0, providers=('exa', 'parallel', 'brave')):
    selected = [r for r in candidates if any(s.provider in providers for s in r.sources)]
    pool = fuse(selected)
    def ranks(result):
        values = {}
        for source in result.sources:
            if source.provider in providers:
                values[source.provider] = min(source.rank, values.get(source.provider, source.rank))
        return values
    exa = sorted([r for r in pool if 'exa' in ranks(r)], key=lambda r: ranks(r)['exa'])
    locked = exa[:keep]
    others = [p for p in providers if p != 'exa']
    def score(result):
        positions = ranks(result)
        prior = max(0, 11 - positions.get('exa', 11))
        support = sum(max(0, 11 - positions.get(p, 11)) for p in others) / max(1, len(others))
        return prior + weight * support
    locked_ids = {r.id for r in locked}
    rest = sorted([r for r in pool if r.id not in locked_ids], key=lambda r: (-score(r), r.id))
    return locked + rest
