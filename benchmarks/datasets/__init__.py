from dataclasses import dataclass, field
import json
from pathlib import Path
from slingsearch.models import Result


@dataclass
class Query:
    id: str
    question: str
    mode: str = 'web'
    candidates: list[Result] = field(default_factory=list)
    qrels: dict[str, int] = field(default_factory=dict)
    provenance: str = ''
    retrieval: dict = field(default_factory=dict)


def rows(path):
    path = Path(path)
    if path.suffix == '.jsonl':
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    data = json.loads(path.read_text())
    return data if isinstance(data, list) else [data]


def load_snapshot(path):
    return [Query(**{**r, 'candidates': [Result.from_dict(c) for c in r.get('candidates', [])]}) for r in rows(path)]
