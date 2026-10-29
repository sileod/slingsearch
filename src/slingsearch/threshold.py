"""Ordinal relevance gating; native reranker scores are never probabilities."""
from .assessment import assessor


async def qualify(client, config, question, candidates, minimum, ranking_backend, bounded):
    backend = None
    try:
        if ranking_backend in ('jev', 'decider', 'd1'):
            scores = {r.id: r.model_score for r in candidates}
            model = ranking_backend
        elif candidates:
            # Dedicated rerankers and RRF have incompatible scales. Keep their
            # ordering, but judge admission with the common five-level rubric.
            backend = assessor(client, config, len(candidates))
            judged = await bounded(backend.rank(question, candidates))
            scores = {r.id: r.model_score for r in judged}
            model = ','.join(sorted(backend.resolved_models)) or backend.model
        else:
            scores, model = {}, None
        if any(score is None or not 1 <= score <= 5 for score in scores.values()):
            raise ValueError('Relevance scores unavailable')
        kept = [r for r in candidates if scores[r.id] >= minimum]
        detail = {'status':'complete', 'minimum':minimum, 'scale':'ordinal_1_to_5',
                  'model':model, 'scored':len(candidates), 'kept':len(kept),
                  'scores':scores, 'scope':'Model judgments on shortlist excerpts, not calibrated probabilities'}
        return kept, detail, backend
    except Exception as error:
        return [], {'status':'unavailable', 'minimum':minimum, 'error_type':type(error).__name__,
                    'scored':0, 'kept':0}, backend
