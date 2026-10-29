"""Export a deterministic sample for separate manual passage-usefulness assessment."""
import argparse
import csv
import random
from slingsearch import Result
from slingsearch.fusion import fuse
from slingsearch.output import budget_results, evidence_tokens
from slingsearch.passages import select
from .datasets import load_snapshot


def export(queries, path, size=100, seed=0):
    selected = random.Random(seed).sample(queries, min(size, len(queries)))
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['query_id', 'question', 'url', 'passage', 'output_tokens',
                                               'useful_evidence', 'verifiable_at_url', 'notes'])
        writer.writeheader()
        for q in selected:
            results = fuse(q.candidates)
            for r in results:
                r.passages = select(q.question, [r.abstract] if r.abstract else r.passages)
            results = budget_results(results, 5, 1500)
            for r in results:
                for passage in r.passages:
                    writer.writerow({'query_id': q.id, 'question': q.question, 'url': r.url, 'passage': passage,
                                     'output_tokens': evidence_tokens(results)})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('snapshot')
    parser.add_argument('--output', default='passage-assessment.csv')
    parser.add_argument('--size', type=int, default=100)
    args = parser.parse_args()
    export(load_snapshot(args.snapshot), args.output, args.size)


if __name__ == '__main__':
    main()
