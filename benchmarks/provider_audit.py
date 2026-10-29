"""Free provider contribution and RRF subset audit on frozen academic results."""
from copy import deepcopy
import json
from pathlib import Path
from statistics import mean
from benchmarks.datasets import load_snapshot
from benchmarks.academic_eval import gold_aliases, score
from slingsearch.fusion import fuse, normalize_doi
from slingsearch.diversity import reserve_shortlist


def report(root, output):
    root, output = Path(root), Path(output)
    queries = load_snapshot(root/'retrieval.jsonl')
    gold = json.loads((root/'gold-metadata.json').read_text())
    providers = ['exa','parallel','openalex','semantic_scholar','serpapi','papers_with_code','huggingface']
    hits = {p:set() for p in providers}
    for query in queries:
        mapping = gold_aliases(query,gold)
        for result in query.candidates:
            aliases = [result.id]+[k+':'+(normalize_doi(v) if k=='doi' else v) for k,v in result.identifiers.items()]
            if result.doi: aliases.append('doi:'+normalize_doi(result.doi))
            matched = {mapping[a] for a in aliases if a in mapping}
            for source in result.sources:
                hits[source.provider].update((query.id,pid) for pid in matched)
    contributions = {}
    for provider, found in hits.items():
        others = set().union(*(value for name,value in hits.items() if name!=provider))
        contributions[provider] = {'known_positive_query_paper_pairs':len(found), 'unique_pairs':len(found-others),
            'additional_pairs_beyond_exa_openalex':len(found-hits['exa']-hits['openalex']),
            'successful_searches':sum(q.retrieval[provider]['success'] for q in queries), 'total_queries':len(queries)}
    subsets = {'exa':['exa'], 'exa_openalex':['exa','openalex'],
               'previous_default':['exa','openalex','semantic_scholar'],
               'pruned_default':['exa','parallel','openalex'],
               'weighted_scholar':['exa','parallel','openalex','semantic_scholar','serpapi'], 'all_sources':providers}
    comparisons = {}
    for name, selected in subsets.items():
        results = []
        for query in queries:
            docs=[]
            for result in deepcopy(query.candidates):
                result.sources=[s for s in result.sources if s.provider in selected]
                if result.sources: docs.append(result)
            weights = {'semantic_scholar':.5, 'serpapi':.5} if name == 'weighted_scholar' else {}
            ranked=fuse(docs, weights)
            if weights:
                ranked=reserve_shortlist(ranked, 30 if query.id.startswith('litqa2') else 20, weights)
            results.append((query,score(query,ranked,gold)))
        comparisons[name] = {group:{'queries':len(rows), 'metric':metric,
            'mean':mean(values[metric] for _,values in rows)}
            for group,metric in [('sage_short','Hit@10'),('sage_open','WeightedRecall@10'),('litqa2','Recall@30')]
            if (rows := [(q,values) for q,values in results if q.id.startswith(group)])}
    summary={'provider_contributions':contributions,'rrf_subsets':comparisons,
        'decision':'Scholar defaults retain Exa, Parallel, OpenAlex, Semantic Scholar and Google Scholar via SerpAPI. The last two receive half-weight fusion votes and one candidate reservation each, with no final-result quota. This is an exploratory policy, not a validated optimum.',
        'limitations':['Exploratory policy selection on 34 reused queries, not independent validation',
                       'SAGE positives are incomplete; zero observed label contributions does not establish general uselessness',
                       'Shared identity metadata is held constant; query wording and provider result depths are not equal-cost retrieval budgets',
                       'GitHub implementation discovery is separate from paper retrieval and is not assessed by these relevance labels']}
    output.mkdir(parents=True,exist_ok=True)
    (output/'results.json').write_text(json.dumps(summary,indent=2))
    lines=['# Provider contribution audit','',summary['decision'],'',
           '| Provider | Known-positive query-paper pairs | Unique pairs | Search success |', '|---|---:|---:|---:|']
    for p,s in contributions.items():lines.append(f"| {p} | {s['known_positive_query_paper_pairs']} | {s['unique_pairs']} | {s['successful_searches']}/{s['total_queries']} |")
    lines+=['','| RRF provider set | SAGE known-paper Hit@10 | SAGE open WeightedRecall@10 | LitQA Recall@30 |','|---|---:|---:|---:|']
    for name, values in comparisons.items():lines.append(f"| {name} | {values['sage_short']['mean']:.3f} | {values['sage_open']['mean']:.3f} | {values['litqa2']['mean']:.3f} |")
    lines+=['',*['- '+item for item in summary['limitations']], '',
            'Reproduce without API calls: `python -m benchmarks.provider_audit`. Private academic snapshots are required. Only aggregate counts are published, not query-paper ground truth.']
    (output/'README.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(comparisons,indent=2))


if __name__=='__main__':
    report('benchmarks/reports/academic-sources','benchmarks/reports/provider-audit')
