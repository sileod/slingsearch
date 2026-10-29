from dataclasses import dataclass, field
import os

WEB = ('exa', 'parallel', 'brave')
SCHOLAR = ('openalex', 'semantic_scholar', 'exa')
NAMES = ('exa', 'parallel', 'brave', 'tavily', 'firecrawl', 'semantic_scholar', 'openalex')
MODELS = {'jev': '~typesafe/jev-latest', 'd1': '~liquid/d1-ultra-latest', 'decider': 'perplexity/pplx-decider-v1.2', 'voyage': 'voyageai/rerank-3-lite'}


@dataclass
class Config:
    api_keys: dict[str, str] = field(default_factory=lambda: {
        name: os.environ.get(name.upper() + '_API_KEY', '') for name in (*NAMES, 'openrouter')
    }, repr=False)
    web_providers: tuple[str, ...] = WEB
    scholar_providers: tuple[str, ...] = SCHOLAR
    timeout: float = 30.0
    ranking_concurrency: int = 5
    candidate_tokens: int = 400
    ensemble_rankers: tuple[str, ...] = ('jev', 'd1', 'voyage')
    decision_batch_size: int = 1
    decision_evidence_checks: bool = False

    def __post_init__(self):
        if self.timeout <= 0 or self.ranking_concurrency < 1 or self.candidate_tokens < 1:
            raise ValueError('Timeout, concurrency and candidate_tokens must be positive')
        if (len(self.ensemble_rankers) < 2 or len(set(self.ensemble_rankers)) != len(self.ensemble_rankers)
                or any(name not in MODELS for name in self.ensemble_rankers)):
            raise ValueError('ensemble_rankers requires at least two distinct model rankers')
        if type(self.decision_batch_size) is not int or not 1 <= self.decision_batch_size <= 20:
            raise ValueError('decision_batch_size must be an integer from 1 to 20')
        if not isinstance(self.decision_evidence_checks, bool):
            raise ValueError('decision_evidence_checks must be a boolean')
