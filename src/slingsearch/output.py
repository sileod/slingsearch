from copy import deepcopy
from dataclasses import asdict
import json
import math


def tokens(text):
    """Conservative UTF-8 byte/3 estimate; not a model-specific tokenizer."""
    return math.ceil(len(text.encode('utf-8')) / 3)


def markdown(results):
    blocks = []
    for i, r in enumerate(results, 1):
        year = f' ({r.year})' if r.year else ''
        found = ', '.join(f'{s.provider} #{s.rank}' for s in r.sources)
        parts = [f'[{i}] {r.title}{year}', r.url, f'Source: {r.id}', f'Found by: {found}']
        parts.extend(f'\nRelevant passage:\n"{p}"' for p in r.passages)
        blocks.append('\n'.join(parts))
    return '\n\n'.join(blocks)


def evidence_tokens(results):
    return max(tokens(markdown(results)), tokens(json.dumps([asdict(r) for r in results], ensure_ascii=False)))


def budget_results(results, max_results, token_budget):
    # Reserve citation metadata for the selected papers before allocating text.
    # Otherwise the first papers consume the budget and later diverse evidence
    # disappears even when five short excerpts would fit.
    originals = list(results[:max_results])
    selected = []
    for original in originals:
        item = deepcopy(original)
        item.passages = []
        item.abstract = None
        selected.append(item)
    while selected and evidence_tokens(selected) > token_budget:
        selected.pop()
    originals = originals[:len(selected)]
    if not selected:
        return []
    max_passages = max((len(r.passages) for r in originals), default=0)
    for passage_index in range(max_passages):
        eligible = [i for i, r in enumerate(originals) if len(r.passages) > passage_index]
        for position, i in enumerate(eligible):
            before = evidence_tokens(selected)
            allowance = max(0, (token_budget - before) // (len(eligible) - position))
            if allowance < 1:
                continue
            text = originals[i].passages[passage_index]
            # Byte/3 is the same estimator used for the complete evidence payload.
            text = text.encode('utf-8')[:allowance * 3].decode('utf-8', errors='ignore')
            selected[i].passages.append(text)
            while text and (evidence_tokens(selected) > token_budget or evidence_tokens(selected) - before > allowance):
                text = text[:-1]
                selected[i].passages[-1] = text
            if not text:
                selected[i].passages.pop()
    for original, item in zip(originals, selected):
        if original.abstract and item.passages:
            item.abstract = item.passages[0]
            if evidence_tokens(selected) > token_budget:
                item.abstract = None
    return selected
