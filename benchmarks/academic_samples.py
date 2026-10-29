"""Prepare pinned, stratified samples without calling search or model APIs.

python -m benchmarks.academic_samples --output benchmarks/reports/academic-samples
"""
import argparse
import ast
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import httpx
from .datasets import sage, litqa2_search


def prepare(output, *, per_stratum=3, seed=20261029):
    from datasets import load_dataset
    from huggingface_hub import HfApi
    if per_stratum < 1:
        raise ValueError('per_stratum must be positive')
    output = Path(output)
    raw = output / 'raw'
    raw.mkdir(parents=True, exist_ok=True)
    manifest = {'seed': seed, 'per_stratum': per_stratum, 'sources': {},
                'protocol': 'Live API adaptation; not a closed-corpus leaderboard reproduction'}
    rng = random.Random(seed)
    sample = []
    with httpx.Client(timeout=60, follow_redirects=True) as client:
        def github(repo):
            response = client.get(f'https://api.github.com/repos/{repo}/commits/main')
            response.raise_for_status()
            return response.json()['sha']
        def download(repo, revision, name, dest):
            response = client.get(f'https://raw.githubusercontent.com/{repo}/{revision}/{name}')
            response.raise_for_status()
            dest.write_bytes(response.content)
            manifest['sources'][name] = {'revision': revision,
                                         'sha256': hashlib.sha256(response.content).hexdigest()}
            return response.text
        revision = github('HughieHu/Sage')
        for kind, folder in sage.KINDS.items():
            for domain in sage.DOMAINS:
                dest = raw / f'sage_{kind}_{domain}.json'
                download('HughieHu/Sage', revision, f'{folder}/{domain}.json', dest)
                pool = sage.load(dest, kind=kind, domain=domain)
                sample.extend(rng.sample(pool, min(per_stratum, len(pool))))
        revision = github('allenai/asta-bench')
        text = download('allenai/asta-bench', revision,
                        'astabench/evals/paper_finder/paper_finder_utils.py', raw / 'asta_mapping_source.py')
        # Parse only literal assignments; never execute downloaded Python.
        wanted = {'litqa2_ids_mapping', 'litqa2_query_ids_in_vespa_fulltext'}
        literals = {}
        for node in ast.parse(text).body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id in wanted:
                        literals[target.id] = ast.literal_eval(node.value)
        mapping_path, subset_path = raw / 'litqa_mapping.json', raw / 'litqa_subsets.json'
        mapping_path.write_text(json.dumps(literals['litqa2_ids_mapping']))
        subset_path.write_text(json.dumps(literals['litqa2_query_ids_in_vespa_fulltext']))
    hf_revision = HfApi().dataset_info('futurehouse/lab-bench').sha
    dataset = load_dataset('futurehouse/lab-bench', 'LitQA2', split='train', revision=hf_revision)
    questions_path = raw / 'litqa_questions.json'
    questions_path.write_text(json.dumps([dict(row) for row in dataset]))
    manifest['sources']['futurehouse/lab-bench'] = {'revision': hf_revision,
        'sha256': hashlib.sha256(questions_path.read_bytes()).hexdigest()}
    litqa = litqa2_search.load(questions_path, mapping_path, subset_path)
    sample.extend(litqa)  # All ten official validation questions; test split stays untouched.
    manifest['query_ids'] = [q.id for q in sample]
    manifest['counts'] = {'sage': len(sample) - len(litqa), 'litqa2_validation': len(litqa)}
    (output / 'sample.json').write_text(json.dumps([asdict(q) for q in sample], indent=2))
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest['counts']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--per-stratum', type=int, default=3)
    parser.add_argument('--seed', type=int, default=20261029)
    args = parser.parse_args()
    prepare(args.output, per_stratum=args.per_stratum, seed=args.seed)
