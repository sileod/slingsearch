"""Local selection rules: provider coverage before ranking, novelty near score ties."""
import math
import re


def balanced_shortlist(candidates, limit):
    """Round-robin original provider ranks, then fill in fused order.

    A result remains one document with all its provenance. This is a candidate
    coverage rule, not an assertion that every provider deserves a final slot.
    """
    if limit < 1:
        raise ValueError('limit must be positive')
    queues = {}
    for result in candidates:
        for source in result.sources:
            queues.setdefault(source.provider, []).append((source.rank, result.id, result))
    for name in queues:
        queues[name].sort(key=lambda item: (item[0], item[1]))
    names = sorted(queues)
    selected, seen = [], set()
    while len(selected) < limit:
        added = False
        for name in names:
            queue = queues[name]
            while queue and queue[0][1] in seen:
                queue.pop(0)
            if queue and len(selected) < limit:
                _, identifier, result = queue.pop(0)
                selected.append(result)
                seen.add(identifier)
                added = True
        if not added:
            break
    for result in candidates:
        if len(selected) >= limit:
            break
        if result.id not in seen:
            selected.append(result)
            seen.add(result.id)
    # The relevance ranker receives a stable RRF ordering, irrespective of which
    # provider first secured a candidate's slot.
    positions = {result.id: i for i, result in enumerate(candidates)}
    return sorted(selected, key=lambda result: positions[result.id])


_STOP = {'the','and','for','with','from','this','that','are','was','were','have','has',
         'using','paper','study','research','present','propose','method','approach'}


def _terms(result):
    text = result.title + ' ' + ' '.join(result.passages)
    return set(re.findall(r'\w{3,}', text.casefold())) - _STOP


def diverse_order(results, *, band_fraction=.05):
    """Prefer different content/providers only within narrow relevance bands.

    Scores are normalized within this query/backend, never across models.
    The highest-ranked document stays first. There are no final provider quotas.
    """
    if not 0 < band_fraction <= 1:
        raise ValueError('band_fraction must be in (0, 1]')
    if len(results) < 2:
        return list(results)
    values = [r.model_score if r.model_score is not None else r.rrf_score for r in results]
    if not all(math.isfinite(v) for v in values):
        raise ValueError('Scores must be finite')
    high, low = max(values), min(values)
    width = (high - low) * band_fraction
    bands = [int((high - value) / width + 1e-9) if width else 0 for value in values]
    terms = [_terms(r) for r in results]
    chosen, remaining = [0], list(range(1, len(results)))
    seen_providers = {s.provider for s in results[0].sources}
    while remaining:
        def key(index):
            overlaps = [len(terms[index] & terms[j]) / max(1, len(terms[index] | terms[j])) for j in chosen]
            new_provider = bool({s.provider for s in results[index].sources} - seen_providers)
            return bands[index], max(overlaps), -int(new_provider), index
        index = min(remaining, key=key)
        chosen.append(index)
        remaining.remove(index)
        seen_providers.update(s.provider for s in results[index].sources)
    return [results[i] for i in chosen]


def preserve_provider_first(results, provider):
    """Keep the best available original result of an explicitly chosen provider.

    This is an exposed retrieval prior, not a change to model scores. Remaining
    papers keep the global relevance/diversity ordering and all provenance.
    """
    candidates = [r for r in results if any(s.provider == provider for s in r.sources)]
    if not candidates:
        return list(results)
    first = min(candidates, key=lambda r: (min(s.rank for s in r.sources if s.provider == provider), r.id))
    return [first] + [r for r in results if r.id != first.id]


def reserve_shortlist(candidates, limit, providers=()):
    """Reserve at most one original top-ranked candidate per named provider.

    Reservations affect candidate admission only, never final result quotas.
    Empty provider lists preserve the deterministic fused baseline.
    """
    if limit <= 0:
        return []
    selected = set()
    for provider in providers:
        available = [(min(s.rank for s in r.sources if s.provider == provider), i)
                     for i, r in enumerate(candidates) if any(s.provider == provider for s in r.sources)]
        if available and len(selected) < limit:
            selected.add(min(available)[1])
    for i in range(len(candidates)):
        if len(selected) >= limit:
            break
        selected.add(i)
    return [r for i, r in enumerate(candidates) if i in selected]
