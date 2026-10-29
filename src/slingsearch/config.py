from dataclasses import dataclass, field
import os

WEB = ('exa', 'parallel', 'brave')
SCHOLAR = ('openalex', 'semantic_scholar', 'exa')
NAMES = ('exa', 'parallel', 'brave', 'tavily', 'firecrawl', 'semantic_scholar', 'openalex')
MODELS = {'jev': 'typesafe/jev-1.14', 'decider': 'perplexity/pplx-decider-v1.2', 'voyage': 'voyageai/rerank-3-lite'}


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

    def __post_init__(self):
        if self.timeout <= 0 or self.ranking_concurrency < 1 or self.candidate_tokens < 1:
            raise ValueError('Timeout, concurrency and candidate_tokens must be positive')
