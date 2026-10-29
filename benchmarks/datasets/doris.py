"""Read released DORIS-MAE JSON without loading its gigabyte corpus into RAM.

Labels are released AnnoGPT aspect judgments, not independent human gold labels.
The summed aspect gain matches the upstream compute_relevance_score function.
"""
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from . import Query
from slingsearch.models import Result


@dataclass
class Case:
    query: Query
    aspect_scores: dict[str, dict[str, int]] = field(default_factory=dict)
    aspect_text: dict[str, str] = field(default_factory=dict)


def load(path, indices):
    import ijson
    path = Path(path)
    with path.open('rb') as f:
        queries = list(ijson.items(f, 'Query.item'))
    if len(set(indices)) != len(indices) or any(i < 0 or i >= len(queries) for i in indices):
        raise ValueError('Invalid or duplicate DORIS query indices')
    selected = [queries[i] for i in indices]
    wanted = {int(d) for q in selected for d in q['candidate_pool']}
    with path.open('rb') as f:
        papers = {int(p['abstract_id']): p for p in ijson.items(f, 'Corpus.item')
                  if int(p['abstract_id']) in wanted}
    if wanted - papers.keys():
        raise ValueError('Missing candidate abstracts')
    aspects = {str(a) for q in selected for a, children in q['aspects'].items()
               for a in [a, *children]}
    annotations = {}
    with path.open('rb') as f:
        for row in ijson.items(f, 'Annotation.item'):
            key = (str(row['aspect_id']), int(row['abstract_id']))
            if key[0] in aspects and key[1] in wanted:
                score = int(row['score'])
                if score not in (0, 1, 2):
                    raise ValueError('Invalid or unresolved DORIS judgment')
                if key in annotations and annotations[key] != score:
                    raise ValueError('Conflicting DORIS judgments')
                annotations[key] = score
    with path.open('rb') as f:
        texts = dict(ijson.kvitems(f, 'aspect_id2aspect'))
    result = []
    for index, q in zip(indices, selected):
        aid = sorted({str(a) for parent, children in q['aspects'].items() for a in [parent, *children]})
        scores, candidates, qrels = {}, [], {}
        for pid in q['candidate_pool']:
            doc = f'doris:{pid}'
            if any((a, pid) not in annotations for a in aid):
                raise ValueError(f'Incomplete judgments for query {index}, abstract {pid}')
            scores[doc] = {a: annotations[a, pid] for a in aid}
            qrels[doc] = sum(scores[doc].values())
            paper = papers[pid]
            abstract = paper['original_abstract']
            candidates.append(Result(doc, paper['title'].replace('_', ' '), paper['url'],
                                     passages=[abstract], abstract=abstract,
                                     identifiers={'semantic_scholar': paper['ss_id']} if paper.get('ss_id') else {}))
        query = Query(f'doris-{index}', q['query_text'], 'scholar', candidates, qrels,
                      'DORIS-MAE v1, Zenodo 8299749; summed released AnnoGPT judgments')
        result.append(Case(query, scores, {a: texts[a] for a in aid}))
    return result


def save_cases(cases, path):
    Path(path).write_text(json.dumps([asdict(c) for c in cases], ensure_ascii=False))


def load_cases(path):
    result = []
    for c in json.loads(Path(path).read_text()):
        q = c['query']
        q['candidates'] = [Result.from_dict(r) for r in q['candidates']]
        result.append(Case(Query(**q), c['aspect_scores'], c['aspect_text']))
    return result
