"""Ordinal relevance gating; native reranker scores are never probabilities."""
import math
from .assessment import assessor


def validate_min_relevance(value):
    if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))
                              or not math.isfinite(value) or not 0 <= value <= 1):
        raise ValueError("min_relevance must be a finite number between zero and one, or None")


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
        raw_scores = scores
        scores = {identifier: (score - 1) / 4 for identifier, score in raw_scores.items()}
        for result in candidates:
            result.relevance_score = scores[result.id]
        kept = [r for r in candidates if scores[r.id] >= minimum]
        detail = {'status':'complete', 'minimum':minimum, 'scale':'normalized_0_to_1',
                  'model':model, 'scored':len(candidates), 'kept':len(kept),
                  'scores':scores, 'raw_scores':raw_scores, 'normalization':'(rubric_score - 1) / 4', 'scope':'Model judgments on shortlist excerpts, not calibrated probabilities'}
        return kept, detail, backend
    except Exception as error:
        return [], {'status':'unavailable', 'minimum':minimum, 'error_type':type(error).__name__,
                    'scored':0, 'kept':0}, backend
