"""Join LAB-bench questions to AstaBench's official FullText-Search subset.

The mapping and subset are literal exports from paper_finder_utils.py, not
hand-picked questions. No ideal answers, choices or key passages enter queries.
"""
import json
from pathlib import Path
from . import Query, rows


def load(questions_path, mapping_path, subset_path, *, split='validation'):
    if split not in ('validation', 'test'):
        raise ValueError('Use an official validation or test split')
    mapping = json.loads(Path(mapping_path).read_text())
    subsets = json.loads(Path(subset_path).read_text())
    selected = set(subsets[split])
    queries, found = [], set()
    for row in rows(questions_path):
        item = mapping.get(str(row['id']))
        if item is None or item['query_id'] not in selected:
            continue
        qid = item['query_id']
        if qid in found:
            raise ValueError('Duplicate LitQA2 question')
        qrels = {'corpusid:' + str(pid): 1 for pid in item['corpus_ids']}
        if not qrels or not row['question'].strip():
            raise ValueError('LitQA2 requires a question and target paper')
        queries.append(Query(qid, row['question'], 'scholar', qrels=qrels,
                             provenance='AstaBench LitQA2-FullText-Search official subset',
                             retrieval={'split': split, 'inserted_before': '2024-10-17',
                                        'primary_metric': 'Recall@30'}))
        found.add(qid)
    if found != selected:
        raise ValueError('Missing questions from official LitQA2 subset')
    return queries
