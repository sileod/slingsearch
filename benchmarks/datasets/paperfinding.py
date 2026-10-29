"""Loader for the official gated PaperFindingBench release; no downloading on import."""
from dataclasses import dataclass
from . import Query, rows


@dataclass
class Case:
    query: Query
    criteria: dict
    query_type: str


def load(path):
    cases = []
    for row in rows(path):
        inp, criteria = row['input'], row['scorer_criteria']
        qid = str(inp['query_id'])
        kind = qid.split('_')[0]
        if kind not in ('semantic', 'specific', 'metadata'):
            raise ValueError('Unknown PaperFindingBench query type')
        if kind == 'semantic' and not criteria.get('relevance_criteria'):
            raise ValueError('Semantic questions require released relevance criteria')
        # Semantic known-positive sets are incomplete: do not promote them to exhaustive qrels.
        qrels = {} if kind == 'semantic' else {
            'corpusid:' + str(pid): 1 for pid in criteria['corpus_ids']}
        cases.append(Case(Query(qid, inp['query'], 'scholar', qrels=qrels,
                                provenance='PaperFindingBench official release'), criteria, kind))
    return cases
