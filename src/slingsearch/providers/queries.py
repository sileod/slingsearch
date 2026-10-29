"""Deterministic cleanup for the lexical Semantic Scholar API; no model calls."""
import re

# Remove conversational scaffolding, not scientific entities or constraints.
_STOP = set('''a an the and or of to in on for from with by as at is are was were be been
can could would should you me i my we us our any some there which what where how
find point recommend suggest looking interested please papers paper studies study
research resources resource tools tool methods method approach approaches focused
focus focuses about specifically particularly discusses discussion detailed introduces
introduce provides provide using use uses used systems system large scale recent
works work publications publication literature available explores explore exploring
that this these those it its do does did have has had if so such based well known'''.split())


def scholar_keywords(question, max_terms=10):
    tokens = re.findall(r'[\w]+', question.replace('-', ' '))
    retained = []
    for token in tokens:
        if token.casefold() not in _STOP and token.casefold() not in {t.casefold() for t in retained}:
            retained.append(token)
    if not retained:
        return question.replace('-', ' ')
    # Preserve mixed-case scientific names, acronyms and dataset identifiers.
    important = [t for t in retained if any(c.isupper() for c in t[1:]) or any(c.isdigit() for c in t)]
    selected = set(t.casefold() for t in important[:max_terms])
    for token in retained:
        if len(selected) >= max_terms:
            break
        selected.add(token.casefold())
    return ' '.join(t for t in retained if t.casefold() in selected)
