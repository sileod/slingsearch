import re
from . import Query, rows
from slingsearch.models import Result, Source


def load(path=None, *, revision=None, grade_field=None):
    if path is None:
        if not revision:
            raise ValueError('Pin a dataset revision for reproducible downloads')
        from datasets import load_dataset
        data = load_dataset('XiaofengAlg/SPARBench', split='train', revision=revision)
    else:
        data = rows(path)
    queries = []
    for i, row in enumerate(data):
        q = Query(str(i), row['query'], 'scholar', provenance='SPARBench: selected candidate papers; labels require explicit field')
        for rank, paper in enumerate(row['answers'], 1):
            pid = paper['paperID']
            arxiv = bool(re.fullmatch(r'\d{4}\.\d{4,5}(?:v\d+)?', pid))
            url = ('https://arxiv.org/abs/' if arxiv else 'https://www.semanticscholar.org/paper/') + pid
            result = Result(pid, paper['title'], url, abstract=paper.get('abstract'),
                            authors=paper.get('author') or [], year=paper.get('year'),
                            identifiers={'arxiv' if arxiv else 'semantic_scholar': pid},
                            sources=[Source('annotated_pool', rank)])
            q.candidates.append(result)
            if grade_field:
                # Do not silently promote released model scores to human ground truth.
                grade = paper[grade_field]
                if not isinstance(grade, int) or isinstance(grade, bool) or grade < 0:
                    raise ValueError('Use explicit nonnegative integer relevance grades, not continuous model scores')
                q.qrels[url] = grade
        queries.append(q)
    return queries
