"""Replay frozen pools and persist per-query runs, metrics and paired comparisons."""
import argparse
import asyncio
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from statistics import mean, median
from time import perf_counter
import httpx
from slingsearch.config import Config
from slingsearch.fusion import fuse
from slingsearch.output import budget_results, markdown, tokens
from slingsearch.passages import representation, select
from slingsearch.rankers import make_ranker
from .datasets import load_snapshot
from .evaluate import bootstrap_delta, document_id, metrics


def percentile(values, fraction):
    return sorted(values)[min(len(values) - 1, int((len(values) - 1) * fraction))]


async def compare(queries, *, methods=('rrf', 'jev', 'decider', 'voyage'), config=None, shortlist=20, transport=None):
    if not 1 <= shortlist <= 50:
        raise ValueError('shortlist must be between 1 and 50')
    config = config or Config()
    records = []
    async with httpx.AsyncClient(timeout=config.timeout, transport=transport) as client:
        for query in queries:
            if not query.candidates:
                raise ValueError(f'{query.id}: no frozen candidates; collect live outputs first')
            pool = fuse(query.candidates)[:shortlist]
            for r in pool:
                r.passages = select(query.question, [r.abstract] if query.mode == 'scholar' and r.abstract else r.passages)
            texts = [representation(r, config.candidate_tokens) for r in pool]
            digest = hashlib.sha256(json.dumps([query.question, [r.id for r in pool], texts]).encode()).hexdigest()
            runs = {}
            providers = sorted({s.provider for r in query.candidates for s in r.sources} | set(query.retrieval))
            for provider in providers:
                # Resolve document identity once across the full pool so richer
                # metadata from another provider cannot inflate ensemble recall.
                found = [r for r in fuse(query.candidates) if any(s.provider == provider for s in r.sources)]
                runs['single:' + provider] = sorted(found, key=lambda r: min(s.rank for s in r.sources if s.provider == provider))
            for method in [*runs, *methods]:
                start = perf_counter()
                ranker, error = None, None
                try:
                    if method in runs:
                        ranked = runs[method]
                    else:
                        ranker = make_ranker(method, client, config)
                        ranked = await ranker.rank(query.question, deepcopy(pool))
                except Exception as exc:
                    ranked, error = [], type(exc).__name__
                ranking_latency = perf_counter() - start
                retrieval = query.retrieval.get(method.removeprefix('single:'), {}) if method.startswith('single:') else {}
                record = {'query_id': query.id, 'mode': query.mode, 'method': method, 'pool_hash': digest,
                          'success': error is None and retrieval.get('success', True), 'error': error, 'ranking_latency': ranking_latency,
                          'latency': retrieval.get('latency') if method.startswith('single:') else ranking_latency,
                          'latency_scope': 'retrieval' if method.startswith('single:') else 'offline_ranking',
                          'cost': getattr(ranker, 'cost_usd', None), 'usage': getattr(ranker, 'usage', {}),
                          'model': sorted(getattr(ranker, 'resolved_models', [])),
                          'metrics': metrics(ranked, query.qrels) if error is None else {},
                          'result_ids': [r.id for r in ranked], 'provenance': query.provenance,
                          'unique_relevant_contributions': {p: sum(
                              query.qrels.get(document_id(r, query.qrels), 0) > 0
                              and {s.provider for s in r.sources} == {p} for r in fuse(query.candidates))
                              for p in providers},
                          'output_tokens': tokens(markdown(budget_results(ranked, 5, 1500)))}
                if method.startswith('single:'):
                    record['cost'] = retrieval.get('cost')
                records.append(record)
    return records


def report(records):
    lines = ['Offline replay: model rows measure ranking latency/cost only; single rows use released retrieval telemetry.',
             'Unknown costs/timings are shown as NA. Unjudged documents count as nonrelevant. Best single is selected by aggregate nDCG@5 on this evaluation set (optimistic).']
    for mode in ('web', 'scholar'):
        subset = [r for r in records if r['mode'] == mode]
        if not subset:
            continue
        singles = sorted({r['method'] for r in subset if r['method'].startswith('single:')})
        def avg(method, metric):
            values = [r['metrics'][metric] for r in subset if r['method'] == method and metric in r['metrics']]
            return mean(values) if values else None
        available = [m for m in singles if avg(m, 'nDCG@5') is not None]
        best = max(available, key=lambda m: avg(m, 'nDCG@5')) if available else None
        lines += ['', mode, '', '| Method | nDCG@5 | Recall@10 | Latency p50 (s) | Cost/1000 queries | Success |',
                  '|---|---:|---:|---:|---:|---:|']
        for method in ([best] if best else []) + ['rrf', 'jev', 'decider', 'voyage']:
            rows = [r for r in subset if r['method'] == method]
            values = [r['latency'] for r in rows if r['latency'] is not None]
            costs = [r['cost'] for r in rows if r['cost'] is not None]
            cost = mean(costs) * 1000 if costs and len(costs) == len(rows) else (0 if method == 'rrf' else None)
            display = lambda v: 'NA' if v is None else f'{v:.4f}'
            label = f'Best single ({method[7:]})' if method == best else method
            success = sum(r['success'] for r in rows) / len(rows) if rows else 0
            lines.append('| ' + ' | '.join([label, display(avg(method, 'nDCG@5')), display(avg(method, 'R@10')),
                                           display(median(values) if values else None), display(cost), display(success)]) + ' |')
    return '\n'.join(lines) + '\n'


def paired(records):
    result = {}
    for mode in ('web', 'scholar'):
        for method in ('jev', 'decider', 'voyage'):
            for metric in ('nDCG@5', 'R@10'):
                def values(name):
                    return {r['query_id']: r['metrics'][metric] for r in records
                            if r['mode'] == mode and r['method'] == name and metric in r['metrics']}
                left, right = values(method), values('rrf')
                if left and right:
                    result[f'{mode}/{method}-rrf/{metric}'] = bootstrap_delta(left, right)
    return result


def operational(records):
    summary = {}
    for mode in ('web', 'scholar'):
        for method in sorted({r['method'] for r in records}):
            rows = [r for r in records if r['mode'] == mode and r['method'] == method]
            if not rows:
                continue
            latency = [r['latency'] for r in rows if r['latency'] is not None]
            summary[f'{mode}/{method}'] = {
                'latency_p50': median(latency) if latency else None,
                'latency_p95': percentile(latency, .95) if latency else None,
                'success_rate': mean(r['success'] for r in rows),
                'mean_output_tokens': mean(r['output_tokens'] for r in rows),
                'mean_ranker_tokens': mean(r['usage'].get('approximate_input_tokens', 0) for r in rows),
                'known_cost_queries': sum(r['cost'] is not None for r in rows),
            }
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('snapshot')
    parser.add_argument('--output', default='comparison')
    parser.add_argument('--methods', nargs='+', choices=['rrf', 'jev', 'decider', 'voyage'], default=['rrf'])
    parser.add_argument('--shortlist', type=int, default=20)
    parser.add_argument('--candidate-tokens', type=int, default=400)
    parser.add_argument('--ranking-concurrency', type=int, default=5)
    args = parser.parse_args()
    config = Config(candidate_tokens=args.candidate_tokens, ranking_concurrency=args.ranking_concurrency)
    queries = load_snapshot(args.snapshot)
    records = asyncio.run(compare(queries, methods=args.methods, config=config, shortlist=args.shortlist))
    Path(args.output + '.json').write_text(json.dumps({'records': records, 'paired': paired(records), 'operational': operational(records), 'input_sha256': hashlib.sha256(Path(args.snapshot).read_bytes()).hexdigest(), 'candidate_tokens': config.candidate_tokens, 'shortlist': args.shortlist}, indent=2))
    Path(args.output + '.md').write_text(report(records))
    print(report(records))


if __name__ == '__main__':
    main()
