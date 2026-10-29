from . import Query, rows


def load(path=None, *, revision=None):
    if path is None:
        if not revision:
            raise ValueError('Pin a dataset revision for reproducible downloads')
        from datasets import load_dataset
        data = load_dataset('princeton-nlp/LitSearch', 'query', split='full', revision=revision)
    else:
        data = rows(path)
    return [Query(str(i), r['query'], 'scholar', qrels={f'corpusid:{p}': 1 for p in r['corpusids']},
                  provenance='LitSearch: known relevant corpus IDs; no frozen provider outputs')
            for i, r in enumerate(data)]
