from dataclasses import dataclass, field
import os
import math

WEB = ('exa', 'parallel', 'brave')
SCHOLAR = ('exa', 'parallel', 'openalex', 'semantic_scholar', 'serpapi')
NAMES = ('exa', 'parallel', 'brave', 'tavily', 'firecrawl', 'semantic_scholar', 'openalex', 'serpapi', 'github', 'papers_with_code', 'huggingface')
MODELS = {'jev': '~typesafe/jev-latest', 'd1': '~liquid/d1-ultra-latest', 'decider': 'perplexity/pplx-decider-v1.2', 'voyage': 'voyageai/rerank-3-lite'}


def environment_keys():
    keys = {name: os.environ.get(name.upper() + '_API_KEY', '') for name in (*NAMES, 'openrouter')}
    keys['github'] = os.environ.get('GITHUB_TOKEN') or keys['github']
    keys['huggingface'] = os.environ.get('HF_TOKEN') or keys['huggingface']
    return keys


@dataclass
class Config:
    api_keys: dict[str, str] = field(default_factory=environment_keys, repr=False)
    web_providers: tuple[str, ...] = WEB
    scholar_providers: tuple[str, ...] = SCHOLAR
    scholar_provider_weights: dict[str, float] = field(default_factory=lambda: {'semantic_scholar': .5, 'serpapi': .5})
    huggingface_resource: str = 'papers'
    timeout: float = 30.0
    ranking_concurrency: int = 5
    candidate_tokens: int = 400
    ensemble_rankers: tuple[str, ...] = ('jev', 'd1', 'voyage')
    decision_batch_size: int = 1
    decision_evidence_checks: bool = False

    def __post_init__(self):
        if any(name not in NAMES or not isinstance(weight, (int, float)) or not math.isfinite(weight) or weight <= 0
               for name, weight in self.scholar_provider_weights.items()):
            raise ValueError("Scholar provider weights must be finite positive values for known providers")
        if self.huggingface_resource not in ('papers', 'models', 'datasets'):
            raise ValueError('huggingface_resource must be papers, models or datasets')
        if self.timeout <= 0 or self.ranking_concurrency < 1 or self.candidate_tokens < 1:
            raise ValueError('Timeout, concurrency and candidate_tokens must be positive')
        if (len(self.ensemble_rankers) < 2 or len(set(self.ensemble_rankers)) != len(self.ensemble_rankers)
                or any(name not in MODELS for name in self.ensemble_rankers)):
            raise ValueError('ensemble_rankers requires at least two distinct model rankers')
        if type(self.decision_batch_size) is not int or not 1 <= self.decision_batch_size <= 20:
            raise ValueError('decision_batch_size must be an integer from 1 to 20')
        if not isinstance(self.decision_evidence_checks, bool):
            raise ValueError('decision_evidence_checks must be a boolean')
