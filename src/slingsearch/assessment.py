"""Independent packed Jev judgments of the evidence actually returned to the caller."""
from .config import MODELS
from .rankers.decisions import Decisions


def assessor(client, config, count):
    key = config.api_keys.get('openrouter')
    if not key:
        raise ValueError('OPENROUTER_API_KEY is missing')
    return Decisions(client, key, MODELS['jev'], concurrency=1, text_budget=config.candidate_tokens,
                     batch_size=min(20, max(1, count)), evidence_checks=True)


def summarize(backend, judged, threshold):
    assessments = []
    for result in judged:
        detail = backend.judgments[result.id]
        assessments.append({'id': result.id, 'relevance_score': result.model_score,
                            'evidence_probability': detail['has_answer_evidence'],
                            'needs_more_text_probability': detail['needs_more_text']})
    maximum = max((r['evidence_probability'] for r in assessments), default=None)
    return {'status': 'complete', 'model': ','.join(sorted(backend.resolved_models)) or backend.model,
            'threshold': threshold, 'results': assessments, 'max_evidence_probability': maximum,
            'useful_result_count': sum(r['evidence_probability'] >= threshold and r['relevance_score'] >= 3 for r in assessments),
            'has_potential_evidence': any(r['evidence_probability'] >= threshold and r['relevance_score'] >= 3 for r in assessments),
            'scope': 'Model estimates on displayed excerpts; not calibrated truth, completeness, or probability of search success'}
