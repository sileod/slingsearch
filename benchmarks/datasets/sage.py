"""Official SAGE query files. Labels are reference-derived, not exhaustive qrels."""
import json
from pathlib import Path
from . import Query

DOMAINS = ('computer_science', 'healthcare', 'humanities', 'natural_science')
KINDS = {'short': 'Sage_Short_Form_Questions', 'open': 'Sage_Open_Ended_Questions'}


def load(path, *, kind, domain):
    if kind not in KINDS or domain not in DOMAINS:
        raise ValueError('Unknown SAGE task or domain')
    data = json.loads(Path(path).read_text())
    records = data if kind == 'short' else data['questions']
    queries = []
    for i, row in enumerate(records):
        labels = [(row.get('ground_truth') or {'paperId': row['paper_id'], 'title': row['paper_title']}, 1)] if kind == 'short' else [
            (paper, grade) for group, grade in [('most_relevant', 2), ('relevant', 1)]
            for paper in row['ground_truth'][group]]
        qrels, papers = {}, {}
        for paper, grade in labels:
            pid = 'semantic_scholar:' + str(paper['paperId'])
            qrels[pid] = max(grade, qrels.get(pid, 0))
            papers[pid] = {'title': paper['title'], 'identifiers': {'semantic_scholar': str(paper['paperId'])}}
        question = row['complete_query'] if kind == 'short' else row['question']
        if not question.strip() or not qrels:
            raise ValueError('SAGE requires a question and reference papers')
        queries.append(Query(f'sage_{kind}_{domain}_{i}', question, 'scholar', qrels=qrels,
                             provenance='SAGE official query release; reference-derived labels',
                             retrieval={'task': kind, 'domain': domain, 'reference_papers': papers}))
    return queries
