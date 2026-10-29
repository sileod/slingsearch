"""Bounded contribution searches and seeded missing-prior-art discovery."""
from .diversity import reserve_shortlist
import asyncio
from copy import deepcopy
import re
from time import perf_counter
from urllib.parse import quote, urlsplit
from .config import MODELS
from .fusion import canonical_url, equivalent, fuse, normalize, normalize_doi, title_key
from .graph import FIELDS, citation_graph, paper_result
from .models import Result
from .output import tokens
from .passages import select
from .prior_art import OverlapJudge
from .providers import PROVIDERS
from .providers.queries import scholar_keywords
from .providers.semantic_scholar import SemanticScholar
from .rankers import make_ranker
from .review_models import Claim, Finding, PaperSeed, ReviewReport


def normalize_claims(contribution, claims):
    if claims is None: claims = [contribution] if contribution else []
    if isinstance(claims, str): claims = [claims]
    if isinstance(claims, dict): claims = [Claim(str(k), str(v)) for k, v in claims.items()]
    result = [c if isinstance(c, Claim) else Claim(f'c{i + 1}', c) for i, c in enumerate(claims)]
    if any(not isinstance(c.text, str) or not c.text.strip() or not c.id for c in result):
        raise ValueError('Each claim needs a nonempty ID and text')
    if len({c.id for c in result}) != len(result): raise ValueError('Claim IDs must be unique')
    if len(result) > 20: raise ValueError('At most 20 explicit claims are supported')
    return result


def seed_input(value):
    if isinstance(value, Result): return deepcopy(value)
    if isinstance(value, dict): value = PaperSeed(**value)
    if isinstance(value, PaperSeed):
        url = value.url or ('https://doi.org/' + normalize_doi(value.doi) if value.doi else '')
        return Result(url, value.title, url, doi=value.doi, year=value.year,
                      identifiers={str(k).lower():str(v) for k,v in value.identifiers.items() if v is not None}, kind='paper')
    if not isinstance(value, str) or not value.strip(): raise ValueError('Invalid related-work entry')
    text = value.strip()
    if text.startswith(('https://', 'http://')): return Result(text, '', text, kind='paper')
    if re.match(r'^(?:doi:)?10\.\d{4,9}/', text, re.I):
        doi = normalize_doi(text)
        return Result('doi:' + doi, '', 'https://doi.org/' + doi, doi=doi, kind='paper')
    if re.fullmatch(r'(?:ARXIV:)?\d{4}\.\d{4,5}(?:v\d+)?', text, re.I):
        arxiv = text.split(':')[-1]
        return Result('arxiv:' + arxiv, '', 'https://arxiv.org/abs/' + arxiv, identifiers={'arxiv': arxiv}, kind='paper')
    if re.fullmatch(r'[a-f0-9]{40}', text, re.I) or re.match(r'^(CorpusId|PMID|PMCID|ACL):\S+$', text, re.I):
        key, identifier = text.split(':', 1) if ':' in text else ('semantic_scholar', text)
        return Result(text, '', '', identifiers={key.lower(): identifier}, kind='paper')
    return Result('unresolved:' + title_key(text), text, '', kind='paper')


def graph_id(result):
    if result.identifiers.get('semantic_scholar'): return result.identifiers['semantic_scholar']
    for key, prefix in [('corpusid','CorpusId:'), ('arxiv','ARXIV:'), ('pmid','PMID:'), ('pmcid','PMCID:'), ('acl','ACL:')]:
        if result.identifiers.get(key): return prefix + result.identifiers[key]
    if result.doi or result.identifiers.get('doi'): return 'DOI:' + normalize_doi(result.doi or result.identifiers['doi'])
    if result.url:
        if urlsplit(result.url).hostname in ('doi.org','dx.doi.org'): return 'DOI:' + normalize_doi(result.url)
        if urlsplit(result.url).hostname in ('semanticscholar.org','www.semanticscholar.org'):
            return 'URL:' + result.url
    return None


async def resolve_seed(client, config, seed):
    if seed.url: normalize(seed)
    identifier = graph_id(seed)
    adapter = SemanticScholar(client, config.api_keys.get('semantic_scholar', ''), 'scholar')
    headers = {'x-api-key': adapter.key} if adapter.key else {}
    try:
        if identifier:
            row = await adapter.request('GET', 'https://api.semanticscholar.org/graph/v1/paper/' + quote(identifier, safe=''),
                                        headers=headers, params={'fields': FIELDS})
            return paper_result(adapter, row, 1), None
        if seed.title:
            candidates = await adapter.search(seed.title, 5)
            matches = fuse([r for r in candidates if title_key(r.title) == title_key(seed.title)
                            and (seed.year is None or r.year == seed.year)])
            if len(matches) == 1: return matches[0], None
            return seed, 'Title match ambiguous or unavailable; provide DOI/arXiv/CorpusId'
        return seed, 'No resolvable scholarly identifier'
    except Exception as error:
        return seed, type(error).__name__


async def resolve_seeds(client, config, inputs, title_limit):
    """Batch stable IDs for the entire bibliography; graph expansion has its own cap."""
    adapter = SemanticScholar(client, config.api_keys.get('semantic_scholar', ''), 'scholar')
    headers = {'x-api-key': adapter.key} if adapter.key else {}
    for seed in inputs:
        if seed.url: normalize(seed)
    ids = list(dict.fromkeys(identifier for seed in inputs if (identifier := graph_id(seed))))
    resolved, errors = {}, {}
    for start in range(0, len(ids), 100):
        batch = ids[start:start + 100]
        try:
            rows = await adapter.request('POST', 'https://api.semanticscholar.org/graph/v1/paper/batch',
                headers=headers, params={'fields':FIELDS}, json={'ids':batch})
            if not isinstance(rows, list) or len(rows) != len(batch): raise ValueError('Invalid seed batch mapping')
            for identifier, row in zip(batch, rows):
                if row and row.get('paperId'): resolved[identifier] = paper_result(adapter, row, 1)
                else: errors[identifier] = 'Paper identifier not resolved'
        except Exception as error:
            errors.update({identifier:type(error).__name__ for identifier in batch})
    output = []
    title_lookups = 0
    for seed in inputs:
        identifier = graph_id(seed)
        if identifier:
            output.append((resolved.get(identifier,seed), errors.get(identifier)))
        elif title_lookups < title_limit:
            output.append(await resolve_seed(client,config,seed))
            title_lookups += 1
        else:
            output.append((seed,'Title lookup limit reached; provide a stable paper identifier'))
    return output


def plan_queries(contribution, claims, seeds, queries, limit, *, seeded=False):
    if queries is not None:
        if isinstance(queries, str): queries = [queries]
        if any(not isinstance(q, str) or not q.strip() for q in queries): raise ValueError('Queries must be nonempty strings')
        proposals = [(q, 'explicit', []) for q in queries]
    else:
        proposals = ([(contribution, 'combined_contribution', [c.id for c in claims])] if contribution else [])
        proposals += [(c.text, 'claim', [c.id]) for c in claims]
        proposals += [(c.search_terms or scholar_keywords(c.text, max_terms=8), 'claim_terms', [c.id]) for c in claims]
        # Seed titles are intentionally used as discovery anchors, never treated as findings about the contribution.
        seed_queries = [(seed.title, 'seed_title', []) for seed in seeds if seed.title]
        proposals = (seed_queries[:1] + proposals[:1] + seed_queries[1:] + proposals[1:]) if seeded and contribution else seed_queries + proposals if seeded else proposals + seed_queries
        if not proposals and contribution: proposals = [(contribution, 'contribution', [])]
    seen, result = set(), []
    for text, kind, claim_ids in proposals:
        if text.strip() in seen: continue
        seen.add(text.strip())
        result.append({'id': f'q{len(result) + 1}', 'question': text.strip(), 'kind': kind, 'claim_ids': claim_ids})
        if len(result) == limit: break
    return result


def report_tokens(report):
    # The evidence budget covers findings, judgments and discovery trails, not input/operational diagnostics.
    import json
    from dataclasses import asdict
    return max(tokens(json.dumps([asdict(f) for f in report.findings], ensure_ascii=False)),
               tokens('\n'.join(_finding_markdown(f, i) for i, f in enumerate(report.findings, 1))))


def _finding_markdown(finding, i):
    temp = ReviewReport('evidence', None, [], [], findings=[finding])
    text = temp.to_markdown()
    return text[text.index('[1]'):].replace('[1]', f'[{i}]', 1)


def bound_findings(report, token_budget):
    # Preserve at least ~160 tokens when the source supplies them. If metadata
    # and claim judgments leave too little space, remove a finding, not its evidence.
    minimum = {}
    for finding in report.findings:
        finding.paper.abstract = None
        minimum[id(finding)] = min(160, tokens(''.join(finding.paper.passages)))
    while report.findings and report_tokens(report) > token_budget:
        candidates = []
        for finding in report.findings:
            excess = tokens(''.join(finding.paper.passages)) - minimum[id(finding)]
            if excess <= 0: continue
            for i, passage in enumerate(finding.paper.passages):
                if passage: candidates.append((len(passage), finding, i, excess))
        if candidates:
            _, finding, i, excess = max(candidates, key=lambda item: item[0])
            passage = finding.paper.passages[i]
            trim = min(max(1, len(passage) // 4), max(1, excess))
            while trim > 1:
                proposed = finding.paper.passages.copy()
                proposed[i] = passage[:-trim]
                if tokens(''.join(proposed)) >= minimum[id(finding)]: break
                trim //= 2
            proposed = finding.paper.passages.copy()
            proposed[i] = passage[:-trim]
            if tokens(''.join(proposed)) < minimum[id(finding)]:
                report.findings.pop()
                continue
            finding.paper.passages = proposed
            if any(j.get('status') != 'reserved' for j in finding.judgments):
                finding.judgments = [{'claim_id': j['claim_id'], 'relation': 'unassessed', 'status': 'excerpt_shortened',
                                      'passage_indices': []} for j in finding.judgments]
            continue
        report.findings.pop()


async def review(client, config, *, workflow, contribution=None, papers=(), claims=None, queries=None,
                 providers=None, ranker='jev', judge='jev', max_queries=8, max_seeds=5, graph_limit=5,
                 max_candidates=20, max_results=5, max_covered=3, per_provider=5, token_budget=3000,
                 year=None, expand_graph=True, min_relevance=3, checks=None, check_threshold=.7):
    from .checks import validate_checks
    validate_checks(checks, check_threshold)
    if min_relevance is not None and (type(min_relevance) is not int or not 1 <= min_relevance <= 5):
        raise ValueError("min_relevance must be an integer from 1 to 5")
    started = perf_counter()
    if workflow not in ('novelty','related_work'): raise ValueError('Unknown review workflow')
    if contribution is not None and (not isinstance(contribution, str) or not contribution.strip()): raise ValueError('Contribution must be nonempty')
    if workflow == 'novelty' and not contribution: raise ValueError('Novelty search requires a contribution')
    if workflow == 'related_work' and not papers: raise ValueError('Related-work search requires papers')
    for name, value, upper in [('max_queries',max_queries,30), ('max_seeds',max_seeds,20), ('graph_limit',graph_limit,100),
                               ('max_candidates',max_candidates,50), ('max_results',max_results,50), ('per_provider',per_provider,50),
                               ('token_budget',token_budget,100000)]:
        if type(value) is not int or not 1 <= value <= upper: raise ValueError(f'Invalid {name}')
    if type(max_covered) is not int or not 0 <= max_covered <= 50: raise ValueError('Invalid max_covered')
    if not isinstance(expand_graph, bool) or judge not in ('jev', None): raise ValueError('Use judge=jev or None and a boolean expand_graph')
    if ranker not in ('rrf', 'ensemble', *MODELS): raise ValueError('Unknown ranker')
    names = list(dict.fromkeys(config.scholar_providers if providers is None else providers))
    if any(name not in PROVIDERS for name in names): raise ValueError('Unknown provider')
    if year is not None:
        if type(year) is int: year = (year, year)
        if len(year) != 2 or any(type(y) is not int for y in year) or not 1 <= year[0] <= year[1] <= 9999: raise ValueError('Invalid year range')
    claim_list = normalize_claims(contribution, claims)
    if claim_list and not contribution: raise ValueError('Claim judgments require a contribution')
    report = ReviewReport(workflow, contribution, claim_list, [])
    # Resolve stable IDs in batches for exclusion; cap title lookups and graph expansion separately.
    seeds = []
    if isinstance(papers, (str, PaperSeed, Result, dict)): papers = [papers]
    inputs = [seed_input(paper) for paper in papers]
    if len(inputs) > 200: raise ValueError('At most 200 related-work entries are supported')
    for i, (seed, error) in enumerate(await resolve_seeds(client, config, inputs, max_seeds)):
        if error: report.unresolved_seeds.append({'entry': i, 'title': seed.title, 'url': seed.url, 'reason': error})
        else: report.resolved_seeds.append({'entry': i, 'id': seed.id, 'title': seed.title, 'url': seed.url})
        seeds.append(seed)
    report.query_plan = plan_queries(contribution, claim_list, seeds, queries, max_queries, seeded=workflow == 'related_work')
    query_question = contribution or 'Find research related to: ' + '; '.join(s.title for s in seeds if s.title)
    if not report.query_plan: report.warnings.append('No query text available; supplied identifiers and graph expansion only')
    raw, origins = [], []
    def add(result, origin):
        if result.kind in ('repository','model','dataset'): return
        if not result.url: return
        result = normalize(deepcopy(result))
        raw.append(result)
        origins.append((deepcopy(result), origin))
    for seed in seeds:
        if seed.url: add(seed, {'kind':'seed', 'seed':seed.id})
    # Sequential query rounds keep per-provider quotas bounded; providers within each round run concurrently.
    async def retrieve(name, plan):
        row = {'query_id':plan['id'], 'provider':name}
        key = config.api_keys.get(name, '')
        if not key and name not in ('openalex','semantic_scholar','papers_with_code','huggingface','github'):
            row['status'] = 'skipped_missing_key'; report.searches.append(row); return []
        adapter = PROVIDERS[name](client, key, 'scholar', year)
        start = perf_counter()
        try:
            docs = await adapter.search(plan['question'], per_provider)
            row.update(status='success', count=len(docs))
            if adapter.cost_usd is not None: report.cost_usd = (report.cost_usd or 0) + adapter.cost_usd
            return docs
        except Exception as error:
            row.update(status='failed', error_type=type(error).__name__); return []
        finally:
            row['latency_s'] = perf_counter() - start; report.searches.append(row)
    for plan in report.query_plan:
        for batch in await asyncio.gather(*(retrieve(name, plan) for name in names)):
            for result in batch: add(result, {'kind':'search', 'query_id':plan['id']})
    if expand_graph:
        for seed in seeds[:max_seeds]:
            identifier = graph_id(seed)
            if not identifier: continue
            graph = await citation_graph(client, config, paper_id=identifier, limit=graph_limit)
            report.graphs.append({'seed':seed.id, 'successful':graph['successful'], 'failed':graph['failed'],
                                  'truncated':graph['truncated'], 'edges':graph['edges']})
            for node in graph['nodes']:
                result = Result.from_dict(node)
                if equivalent(result, seed): continue
                incoming = [edge for edge in graph['edges'] if result.id in (edge['source'],edge['target'])]
                for direction in sorted({'citations' if edge['target'] == graph['seed'] else 'references' for edge in incoming}):
                    add(result, {'kind':'graph', 'seed':seed.id, 'direction':direction})
    pool = fuse(raw, config.scholar_provider_weights)
    if year:
        pool = [r for r in pool if r.year is not None and year[0] <= r.year <= year[1]]
        report.warnings.append('Year filtering excludes candidates without a known publication year; it does not verify priority dates')
    def known(result):
        for seed in seeds:
            if seed.doi and result.doi and normalize_doi(seed.doi) == result.doi: return True
            if any(result.identifiers.get(k) == v for k, v in seed.identifiers.items()): return True
            if seed.url and equivalent(seed, result): return True
        return False
    covered = [r for r in pool if known(r)]
    new = [r for r in pool if not known(r)]
    anchors = covered[:min(max_covered, max_candidates)]
    candidates = anchors + reserve_shortlist(new, max_candidates - len(anchors), config.scholar_provider_weights)
    for result in candidates:
        result.passages = select(query_question, [result.abstract] if result.abstract else result.passages)
    report.usage.update(unique_candidates=len(pool), shortlist=len(candidates), known_candidates=len(covered), new_candidates=len(new),
                        fusion={'k':60, 'weights':config.scholar_provider_weights, 'reserved_providers':list(config.scholar_provider_weights)},
                        graph_seed_limit=max_seeds, query_limit=max_queries, providers=names)
    backend = None
    try:
        if candidates:
            backend = make_ranker(ranker, client, config)
            candidates = await backend.rank(query_question, candidates)
            report.ranking_backend = ranker
            report.model = ','.join(sorted(getattr(backend,'resolved_models',set()))) or backend.model
    except Exception as error:
        report.warnings.append(f'{ranker} ranking unavailable ({type(error).__name__}); RRF order retained')
        candidates.sort(key=lambda r:(-r.rrf_score,r.id))
    finally:
        if backend:
            report.usage['ranking'] = backend.usage
            if backend.cost_usd is not None: report.cost_usd = (report.cost_usd or 0) + backend.cost_usd
    if min_relevance is not None:
        from .threshold import qualify
        async def await_operation(operation):
            return await operation
        candidates, detail, gate = await qualify(client, config, query_question, candidates,
                                                 min_relevance, report.ranking_backend, await_operation)
        if report.ranking_backend in ('jev', 'decider', 'd1'):
            detail['model'] = report.model
        report.usage['relevance_filter'] = detail
        if gate:
            report.usage['relevance_filter_usage'] = gate.usage
            if gate.cost_usd is not None: report.cost_usd = (report.cost_usd or 0) + gate.cost_usd
        if detail['status'] != 'complete':
            report.warnings.append('Relevance cutoff unavailable; no papers qualified')
    if checks:
        from .checks import check_sources
        async def check_operation(operation):
            return await operation
        candidates, detail, check_usage, cost = await check_sources(client, config, query_question, candidates, checks, check_threshold, check_operation)
        report.usage['checks'] = detail
        report.usage['check_usage'] = check_usage
        if cost is not None: report.cost_usd = (report.cost_usd or 0) + cost
        if detail['status'] != 'complete': report.warnings.append('Required source checks unavailable; no papers qualified')
    # Reserve separate slots so known papers cannot crowd all discoveries out of the evidence view.
    discoveries = [r for r in candidates if not known(r)][:max_results]
    existing = [r for r in candidates if known(r)][:max_covered]
    chosen = []
    for i in range(max(len(discoveries), len(existing))):
        if i < len(discoveries): chosen.append(discoveries[i])
        if i < len(existing): chosen.append(existing[i])
    papers_out = deepcopy(chosen)  # The finding-aware budget below reserves judgments and retains substantive excerpts.
    for paper in papers_out:
        trail = []
        for source, origin in origins:
            if equivalent(source, paper) and origin not in trail: trail.append(origin)
        report.findings.append(Finding(paper, known(paper), trail))
    # Bound the evidence BEFORE judging: model decisions refer to precisely the excerpts that are displayed.
    if judge and claim_list:
        for finding in report.findings:
            finding.judgments = [{'claim_id': claim.id, 'relation': 'insufficient_evidence',
                'needs_more_text_probability': .12345678901234567, 'model_confidence': .12345678901234567,
                'status': 'reserved', 'inspection_required': True, 'model_relation': 'insufficient_evidence', 'passage_indices': list(range(len(finding.paper.passages)))} for claim in claim_list]
    bound_findings(report, token_budget)
    for finding in report.findings: finding.judgments = []
    if judge and claim_list and report.findings and config.api_keys.get('openrouter'):
        reviewer = OverlapJudge(client, config.api_keys['openrouter'], text_budget=config.candidate_tokens,
                                batch_size=5, concurrency=min(2, config.ranking_concurrency))
        await reviewer.judge(contribution, claim_list, report.findings)
        report.judgment_model = ','.join(sorted(reviewer.resolved_models)) or MODELS['jev']
        report.usage['overlap'] = reviewer.usage
        if reviewer.cost_usd is not None: report.cost_usd = (report.cost_usd or 0) + reviewer.cost_usd
    else:
        for finding in report.findings:
            finding.judgments = [{'claim_id':claim.id, 'relation':'unassessed', 'status':'not_run', 'passage_indices':[]}
                                for claim in claim_list]
        if judge and claim_list: report.warnings.append('Overlap judgments unavailable without OPENROUTER_API_KEY')
    bound_findings(report, token_budget)
    if any(j['status'] == 'unavailable' for f in report.findings for j in f.judgments):
        report.warnings.append('Some overlap judgments failed; unassessed pairs are not evidence of novelty')
    if any(row['status'] == 'failed' for row in report.searches) or any(g['failed'] for g in report.graphs):
        report.warnings.append('Search or citation-graph coverage is partial; inspect recorded failures')
    if len(covered) > len(report.covered): report.warnings.append('Only a bounded subset of existing related work is displayed')
    report.warnings.append('Labels are model estimates on excerpts, not certified equivalence or a novelty probability. Missing results do not establish absence of prior art.')
    report.usage['output_evidence_tokens'] = report_tokens(report)
    report.timings['total'] = perf_counter() - started
    return report
