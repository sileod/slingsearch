from .rrf import RRF
from .decisions import Decisions
from .rerank import Voyage
from ..config import MODELS


def make_ranker(name, client, config):
    if name == 'rrf':
        return RRF()
    if name == 'ensemble':
        from .ensemble import Ensemble
        return Ensemble({member: make_ranker(member, client, config) for member in config.ensemble_rankers})
    if name not in MODELS:
        raise ValueError(f'Unknown ranker: {name}')
    key = config.api_keys.get('openrouter', '')
    if not key:
        raise ValueError('OPENROUTER_API_KEY is missing')
    if name == 'voyage':
        return Voyage(client, key, MODELS[name], config.candidate_tokens)
    return Decisions(client, key, MODELS[name], config.ranking_concurrency, config.candidate_tokens,
                     batch_size=config.decision_batch_size, evidence_checks=config.decision_evidence_checks)
