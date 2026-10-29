"""D1 and rank-fusion ablations on the existing frozen DORIS shortlist.

python -m benchmarks.d1_eval run  # private key prompt; shared $5 ledger
python -m benchmarks.d1_eval report  # free replay
"""
import asyncio
from dataclasses import asdict
import getpass
import json
from pathlib import Path
from statistics import mean, median
from time import perf_counter
import httpx
from slingsearch import Config
from slingsearch.rankers import make_ranker
from slingsearch.rankers.ensemble import fuse_orders
from benchmarks.datasets import doris
from benchmarks.decision_budget import Transport
from benchmarks.evaluate import bootstrap_delta
from benchmarks.scholar_eval import evaluate

ROOT = Path('benchmarks/reports/decision-audit')


async def run(key):
    cases = doris.load_cases(ROOT / 'doris-frozen.json')
    path = ROOT / 'd1-rankings.jsonl'
    records = [json.loads(s) for s in path.read_text().splitlines()] if path.exists() else []
    complete = {(r['query_id'], r['method']) for r in records if r['success']}
    async with httpx.AsyncClient(transport=Transport(ROOT), timeout=60) as client:
        for case in cases:
            for method, batch in [('d1_single', 1), ('d1_packed5', 5)]:
                if (case.query.id, method) in complete:
                    continue
                config = Config(api_keys={'openrouter': key}, decision_batch_size=batch,
                                candidate_tokens=400, ranking_concurrency=12)
                ranker = make_ranker('d1', client, config)
                record = {'query_id': case.query.id, 'method': method, 'requested_model': ranker.model}
                start = perf_counter()
                try:
                    ranked = await ranker.rank(case.query.question, case.query.candidates)
                    record.update(success=True, result_ids=[r.id for r in ranked],
                                  metrics=evaluate(case, [r.id for r in ranked]),
                                  resolved_models=sorted(ranker.resolved_models))
                except Exception as error:
                    record.update(success=False, error_type=type(error).__name__)
                record.update(usage=ranker.usage, reported_cost_usd=ranker.cost_usd,
                              latency_s=perf_counter() - start)
                with path.open('a') as f:
                    f.write(json.dumps(record) + '\n')
                print(json.dumps({'query': case.query.id, 'method': method,
                                  'success': record['success'], 'latency_s': round(record['latency_s'], 2)}), flush=True)
    report()


def report():
    cases = doris.load_cases(ROOT / 'doris-frozen.json')
    records = {}
    for name in ['doris-rankings.jsonl', 'd1-rankings.jsonl']:
        for line in (ROOT / name).read_text().splitlines():
            row = json.loads(line)
            records[row['query_id'], row['method']] = row
    combos = {'jev_voyage': ['jev_single', 'voyage'], 'd1_voyage': ['d1_single', 'voyage'],
              'jev_d1_voyage': ['jev_single', 'd1_single', 'voyage'],
              'jev_packed_voyage': ['jev_packed5', 'voyage'],
              'd1_packed_voyage': ['d1_packed5', 'voyage'],
              'jev_d1_packed_voyage': ['jev_packed5', 'd1_packed5', 'voyage']}
    for case in cases:
        docs = {d.id: d for d in case.query.candidates}
        for method, members in combos.items():
            if not all(records.get((case.query.id, m), {}).get('success') for m in members):
                continue
            rows = [records[case.query.id, m] for m in members]
            orders = [[docs[pid] for pid in row['result_ids']] for row in rows]
            ranked = fuse_orders(orders, case.query.candidates)
            records[case.query.id, method] = {
                'query_id': case.query.id, 'method': method, 'success': True,
                'metrics': evaluate(case, [r.id for r in ranked]),
                'reported_cost_usd': sum(r['reported_cost_usd'] for r in rows),
                'latency_s': None, # Sequential historical runs cannot measure concurrent ensemble latency.
                'usage': {'input_tokens': sum(r['usage'].get('input_tokens', r['usage'].get('total_tokens', 0)) for r in rows)}}
    methods = ['jev_single', 'jev_packed5', 'voyage', 'd1_single', 'd1_packed5', *combos]
    common = {c.query.id for c in cases if all(records.get((c.query.id, m), {}).get('success') for m in methods)}
    if not common:
        print('No complete paired cases yet; attempts saved.')
        return
    summaries, paired = {}, {}
    for method in methods:
        rows = [records[qid, method] for qid in sorted(common)]
        summaries[method] = {'queries': len(rows), 'nDCG@5': mean(r['metrics']['nDCG@5'] for r in rows),
                             'mean_reported_cost_usd': mean(r['reported_cost_usd'] for r in rows),
                             'mean_input_tokens': mean(r['usage'].get('input_tokens', r['usage'].get('total_tokens', 0)) for r in rows),
                             'latency_p50_s': median(r['latency_s'] for r in rows) if rows[0]['latency_s'] is not None else None}
        if method != 'voyage':
            paired[method] = bootstrap_delta({q: records[q, method]['metrics']['nDCG@5'] for q in common},
                                             {q: records[q, 'voyage']['metrics']['nDCG@5'] for q in common})
    result = {'summary': summaries, 'paired_vs_voyage': paired,
              'limitations': ['Same 12 previously evaluated questions; exploratory, not fresh holdout',
                             'Identical frozen BM25 top20 and 400-token document representations',
                             'Ensembles replay rankings; latency not measured; costs add member costs',
                             'Historical Jev and Voyage runs occurred at a different time than D1',
                             'Rank fusion k=60, equal weights; no score calibration assumed']}
    (ROOT / 'd1-summary.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    import sys
    if sys.argv[1] == 'report':
        report()
    else:
        asyncio.run(run(getpass.getpass('OpenRouter key: ')))
