"""Inputs and citation-backed output for contribution and related-work audits."""
from dataclasses import asdict, dataclass, field
import json
import re
from pathlib import Path
from .models import Result


@dataclass
class Claim:
    id: str
    text: str
    search_terms: str | None = None


@dataclass
class PaperSeed:
    title: str = ''
    url: str | None = None
    doi: str | None = None
    year: int | None = None
    identifiers: dict[str, str] = field(default_factory=dict)


@dataclass
class Finding:
    paper: Result
    known: bool
    origins: list[dict] = field(default_factory=list)
    judgments: list[dict] = field(default_factory=list)


@dataclass
class ReviewReport:
    workflow: str
    contribution: str | None
    claims: list[Claim]
    query_plan: list[dict]
    findings: list[Finding] = field(default_factory=list)
    resolved_seeds: list[dict] = field(default_factory=list)
    unresolved_seeds: list[dict] = field(default_factory=list)
    searches: list[dict] = field(default_factory=list)
    graphs: list[dict] = field(default_factory=list)
    ranking_backend: str = 'rrf'
    model: str | None = None
    judgment_model: str | None = None
    usage: dict = field(default_factory=dict)
    cost_usd: float | None = None
    timings: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def discoveries(self):
        return [finding for finding in self.findings if not finding.known]

    @property
    def covered(self):
        return [finding for finding in self.findings if finding.known]

    def to_dict(self):
        return asdict(self)

    def to_json(self, **kwargs):
        return json.dumps(self.to_dict(), ensure_ascii=False, **kwargs)

    def to_markdown(self):
        lines = ['Prior-art evidence review', '',
                 'This search does not establish novelty, absence of prior work, or exhaustive coverage.', '']
        if self.contribution:
            lines += ['Contribution: ' + self.contribution, '']
        lines += ['Claims:'] + [f'- {claim.id}: {claim.text}' for claim in self.claims]
        for i, finding in enumerate(self.findings, 1):
            paper = finding.paper
            lines += ['', f'[{i}] {paper.title}' + (f' ({paper.year})' if paper.year else ''), paper.url,
                      'Already in related work' if finding.known else 'New candidate for inspection',
                      'Source: ' + paper.id,
                      'Found by: ' + ', '.join(f'{s.provider} #{s.rank}' for s in paper.sources)]
            for judgment in finding.judgments:
                label = judgment.get('relation', 'unassessed')
                refs = ', '.join(f'passage {n + 1}' for n in judgment.get('passage_indices', []))
                lines.append(f"{judgment['claim_id']}: {label}" + (f' ({refs})' if refs else '') + ('; inspect more source text' if judgment.get('inspection_required') else ''))
            for n, passage in enumerate(paper.passages, 1):
                sections = list(dict.fromkeys(m['section'] for m in paper.passage_metadata if m.get('index') == n - 1 and m.get('section')))
                label = f'Passage {n}' + (f" ({'; '.join(sections)})" if sections else '')
                lines += [f'{label}: "{passage}"']
            if paper.code_urls: lines.append('Code: ' + ', '.join(paper.code_urls))
            origins = []
            for origin in finding.origins:
                origins.append('query ' + origin['query_id'] if origin['kind'] == 'search' else
                               origin.get('direction', origin['kind']) + ' from ' + origin.get('seed', paper.id))
            if origins: lines.append('Discovery trail: ' + '; '.join(origins))
        if self.unresolved_seeds:
            lines += ['', f'Unresolved related-work entries: {len(self.unresolved_seeds)} (see JSON details).']
        if self.warnings: lines += ['', 'Limitations:'] + ['- ' + warning for warning in self.warnings]
        return '\n'.join(lines)


def papers_from_markdown(text):
    """Read stable paper links, without treating prose/claims as source evidence."""
    urls = re.findall(r'https?://(?:arxiv\.org/(?:abs|pdf)/[^\s)<>]+|(?:dx\.)?doi\.org/[^\s)<>]+|(?:www\.)?semanticscholar\.org/paper/[^\s)<>]+)', text)
    return [PaperSeed(url=url.rstrip('.,;')) for url in dict.fromkeys(urls)]


def load_papers(path):
    path = Path(path)
    text = path.read_text()
    if path.suffix in ('.md', '.markdown'): return papers_from_markdown(text)
    if path.suffix == '.json':
        data = json.loads(text)
        if not isinstance(data, list): raise ValueError('Paper JSON must be a list')
        return [PaperSeed(**entry) if isinstance(entry, dict) else entry for entry in data]
    if path.suffix == '.jsonl':
        return [PaperSeed(**json.loads(line)) for line in text.splitlines() if line.strip()]
    return [line.strip() for line in text.splitlines() if line.strip() and not line.startswith('#')]
