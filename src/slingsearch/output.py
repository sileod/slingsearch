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
        if r.code_urls: parts.append('Code: ' + ', '.join(r.code_urls))
        if r.tasks: parts.append('Tasks: ' + ', '.join(r.tasks))
        if r.dataset_urls: parts.append('Datasets: ' + ', '.join(r.dataset_urls))
        for index, passage in enumerate(r.passages):
            sections = list(dict.fromkeys(m['section'] for m in r.passage_metadata if m.get('index') == index and m.get('section')))
            label = 'Relevant passage' + (f" ({'; '.join(sections)})" if sections else '')
            parts.append(f'\n{label}:\n"{passage}"')
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
    kept_indices = [[] for _ in selected]
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
            else:
                kept_indices[i].append(passage_index)
    for original, item, indices in zip(originals, selected, kept_indices):
        item.passage_metadata = [{**entry,'index':index}
            for index, source in enumerate(indices) for entry in original.passage_metadata
            if entry.get('index') == source]
        if original.abstract and item.passages:
            abstract_excerpts = [p for p in item.passages if p and p in original.abstract]
            item.abstract = abstract_excerpts[0] if abstract_excerpts else None
            if evidence_tokens(selected) > token_budget:
                item.abstract = None
    return selected
