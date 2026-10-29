"""Metrics use ir-measures; no dataset is downloaded on import."""
import random
from statistics import mean
from slingsearch.fusion import canonical_url, normalize_doi, title_key


def document_id(result, qrels):
    aliases = [result.id, result.url, canonical_url(result.url)]
    if result.doi:
        aliases += ['doi:' + normalize_doi(result.doi)]
    aliases += [f'{k.lower()}:{v}' for k, v in result.identifiers.items()]
    for alias in aliases:
        if alias in qrels:
            return alias
    return result.id


def match_paper(result, corpus):
    """Prefer exact IDs/DOIs; title fallback requires a unique title AND matching year."""
    for pid, paper in corpus.items():
        if result.doi and normalize_doi(result.doi) == normalize_doi(paper.get('doi')):
            return pid
        if any(paper.get('identifiers', {}).get(k) == v for k, v in result.identifiers.items()):
            return pid
    matches = [pid for pid, p in corpus.items() if result.year is not None and p.get('year') == result.year
               and title_key(result.title) == title_key(p.get('title', ''))]
    return matches[0] if len(matches) == 1 else None


def metrics(results, qrels):
    if not qrels:
        return {}
    from ir_measures import Qrel, ScoredDoc, calc_aggregate, nDCG, Recall, RR
    measures = [nDCG@5, nDCG@10, Recall@5, Recall@10, Recall@20, RR@10]
    judgments = [Qrel('q', str(doc), grade) for doc, grade in qrels.items()]
    seen, run = set(), []
    for rank, result in enumerate(results, 1):
        doc = document_id(result, qrels)
        if doc not in seen:
            run.append(ScoredDoc('q', doc, 1 / rank))
            seen.add(doc)
    scores = calc_aggregate(measures, judgments, run)
    return {str(k): float(v) for k, v in scores.items()}


def bootstrap_delta(left, right, *, samples=2000, seed=0):
    common = sorted(set(left) & set(right))
    if not common:
        raise ValueError('No paired queries')
    differences = [left[q] - right[q] for q in common]
    rng = random.Random(seed)
    draws = sorted(mean(rng.choices(differences, k=len(differences))) for _ in range(samples))
    return {'mean_delta': mean(differences), 'ci95': [draws[int(samples * .025)], draws[int(samples * .975)]],
            'queries': len(common), 'seed': seed, 'samples': samples}
