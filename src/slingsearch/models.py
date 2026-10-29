from dataclasses import asdict, dataclass, field
import json


@dataclass(frozen=True)
class Source:
    provider: str
    rank: int


@dataclass
class Result:
    id: str
    title: str
    url: str
    passages: list[str] = field(default_factory=list)
    abstract: str | None = None
    doi: str | None = None
    authors: list[str] = field(default_factory=list)
    year: int | None = None
    sources: list[Source] = field(default_factory=list)
    rrf_score: float = 0.0
    model_score: float | None = None
    relevance_score: float | None = None
    identifiers: dict[str, str] = field(default_factory=dict)
    venue: str | None = None
    citation_count: int | None = None
    related_urls: list[str] = field(default_factory=list)
    kind: str = 'web'
    code_urls: list[str] = field(default_factory=list)
    tasks: list[str] = field(default_factory=list)
    dataset_urls: list[str] = field(default_factory=list)
    passage_metadata: list[dict] = field(default_factory=list)

    @classmethod
    def from_dict(cls, value):
        value = dict(value)
        value['sources'] = [Source(**s) if isinstance(s, dict) else s for s in value.get('sources', [])]
        return cls(**value)


@dataclass
class SearchResponse:
    question: str
    mode: str
    results: list[Result] = field(default_factory=list)
    attempted: list[str] = field(default_factory=list)
    successful: list[str] = field(default_factory=list)
    failed: dict[str, str] = field(default_factory=dict)
    skipped: dict[str, str] = field(default_factory=dict)
    cancelled: dict[str, str] = field(default_factory=dict)
    ranking_backend: str = 'rrf'
    model: str | None = None
    requested_ranker: str = 'jev'
    requested_model: str | None = None
    timings: dict[str, float] = field(default_factory=dict)
    usage: dict = field(default_factory=dict)
    cost_usd: float | None = None
    warnings: list[str] = field(default_factory=list)
    evidence_assessment: dict | None = None

    def to_dict(self):
        return asdict(self)

    def to_json(self, **kwargs):
        return json.dumps(self.to_dict(), ensure_ascii=False, **kwargs)

    def to_markdown(self):
        from .output import markdown
        evidence = markdown(self.results)
        statuses = []
        cutoff = self.usage.get('relevance_filter')
        if cutoff:
            statuses.append(f"Relevance cutoff ≥{cutoff['minimum']:g}: {cutoff['kept']} qualified sources before checks and output limits."
                            if cutoff['status'] == 'complete' else
                            'Relevance scoring unavailable; no sources qualified. This is not evidence that no relevant sources exist.')
        checks = self.usage.get('checks')
        if checks:
            statuses.append(f"Required source checks: {checks['kept']} passed all checks before output limits."
                            if checks['status'] == 'complete' else
                            'Required source checks unavailable; no sources qualified.')
        return '\n\n'.join([*statuses, *([evidence] if evidence else [])])
