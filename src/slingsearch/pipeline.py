import asyncio
import math
from time import perf_counter
from .config import MODELS
from .models import SearchResponse
from .providers import PROVIDERS
from .fusion import fuse
from .passages import extract, select
from .output import budget_results, evidence_tokens, tokens
from .rankers import make_ranker
from .deadline import collect
from .diversity import balanced_shortlist, diverse_order, preserve_provider_first


async def run(client, config, question, *, mode='web', providers=None, ranker='jev',
              max_results=5, token_budget=1500, per_provider=10, shortlist=20,
              passages_per_result=2, fetch_pages=False, year=None, diversity=False, anchor_provider=None,
              fast=False, deadline=None, fast_min_results=None, fast_grace=.15,
              assess=False, assessment_threshold=.7):
    if not isinstance(assess, bool) or isinstance(assessment_threshold, bool) or not isinstance(assessment_threshold, (int, float)) or not math.isfinite(assessment_threshold) or not 0 <= assessment_threshold <= 1:
        raise ValueError('assess must be boolean and assessment_threshold must be between zero and one')
    if not isinstance(fast, bool):
        raise ValueError('fast must be a boolean')
    for name, value in [('deadline', deadline), ('fast_grace', fast_grace)]:
        if name == 'fast_grace' and value is None:
            raise ValueError('fast_grace must be a finite nonnegative number')
        if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0 or name == 'deadline' and value == 0):
            raise ValueError(f'{name} must be finite and positive (grace may be zero)')
    if fast_min_results is not None and (type(fast_min_results) is not int or fast_min_results < 1):
        raise ValueError('fast_min_results must be a positive integer')
    if not isinstance(question, str) or not question.strip():
        raise ValueError('question must be nonempty')
    if mode not in ('web', 'scholar') or ranker not in ('rrf', 'ensemble', *MODELS):
        raise ValueError('Unknown mode or ranker')
    for name, value in {'max_results': max_results, 'token_budget': token_budget, 'per_provider': per_provider,
                        'shortlist': shortlist, 'passages_per_result': passages_per_result}.items():
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f'{name} must be a positive integer')
    if shortlist > 50 or per_provider > 50:
        raise ValueError('shortlist and per_provider cannot exceed 50')
    if year is not None:
        if isinstance(year, int):
            year = (year, year)
        if len(year) != 2 or not all(isinstance(y, int) for y in year) or not 1 <= year[0] <= year[1] <= 9999:
            raise ValueError('year must be a year or an inclusive (start, end) pair')
    if not isinstance(diversity, bool):
        raise ValueError('diversity must be a boolean')
    names = list(dict.fromkeys(providers if providers is not None else
                 config.scholar_providers if mode == 'scholar' else config.web_providers))
    if anchor_provider is not None and anchor_provider not in names:
        raise ValueError('anchor_provider must be one of the selected providers')
    if any(n not in PROVIDERS for n in names):
        raise ValueError('Unknown provider')
    requested_model = ','.join(MODELS[name] for name in config.ensemble_rankers) if ranker == 'ensemble' else MODELS.get(ranker)
    response = SearchResponse(question, mode, requested_ranker=ranker, requested_model=requested_model)
    started = perf_counter()
    if fast and deadline is None:
        deadline = 3.0
    stop_at = started + deadline if deadline is not None else None
    async def bounded(awaitable):
        if stop_at is None:
            return await awaitable
        return await asyncio.wait_for(awaitable, max(0, stop_at - perf_counter()))

    async def retrieve(name):
        key = config.api_keys.get(name, '')
        if not key and name not in ('openalex', 'semantic_scholar', 'github', 'papers_with_code', 'huggingface'):
            response.skipped[name] = 'Missing ' + name.upper() + '_API_KEY'
            response.warnings.append(f'{name}: skipped because credentials are missing')
            return []
        if not key and name != 'papers_with_code':
            response.warnings.append(f'{name}: using keyless API access with lower limits')
        response.attempted.append(name)
        adapter = PROVIDERS[name](client, key, mode, year, **({'resource': config.huggingface_resource} if name == 'huggingface' else {}))
        start = perf_counter()
        try:
            result = await adapter.search(question, per_provider)
            response.successful.append(name)
            if adapter.cost_usd is not None:
                response.cost_usd = (response.cost_usd or 0) + adapter.cost_usd
            return result
        except Exception as error:
            # Never serialize exception text: it can contain keys, URLs or response bodies.
            response.failed[name] = type(error).__name__
            response.warnings.append(f'{name}: retrieval failed ({type(error).__name__})')
            return []
        finally:
            response.timings[name] = perf_counter() - start
    batches = await collect(retrieve, names, response, stop_at=stop_at, fast=fast,
                            min_results=fast_min_results or max_results, grace_s=fast_grace,
                            year=year if mode == 'scholar' else None)
    candidates = fuse([r for batch in batches for r in batch])
    if year and mode == 'scholar':
        candidates = [r for r in candidates if r.year is not None and year[0] <= r.year <= year[1]]
        response.warnings.append('Year filter excludes papers without a known publication year')
    candidates = balanced_shortlist(candidates, shortlist) if diversity else candidates[:shortlist]
    start = perf_counter()
    semaphore = asyncio.Semaphore(5)
    async def prepare(candidate):
        texts = [candidate.abstract] if mode == 'scholar' and candidate.abstract else candidate.passages
        if fetch_pages and mode == 'web' and tokens(''.join(texts)) < 80:
            try:
                async with semaphore:
                    body = await extract(client, candidate.url)
                texts = [*texts, body]
            except Exception as error:
                response.warnings.append(f'Page extraction failed for {candidate.id} ({type(error).__name__})')
        candidate.passages = select(question, texts, passages_per_result)
    try:
        await bounded(asyncio.gather(*(prepare(c) for c in candidates)))
    except TimeoutError:
        response.warnings.append('Deadline reached during passage preparation; using available provider text')
    response.timings['passages'] = perf_counter() - start
    start = perf_counter()
    backend = None
    if candidates:
        if ranker != 'rrf' and not config.api_keys.get('openrouter'):
            response.warnings.append('OPENROUTER_API_KEY is missing; model ranking is disabled')
        try:
            backend = make_ranker(ranker, client, config)
            candidates = await bounded(backend.rank(question, candidates))
            response.ranking_backend = ranker
            response.model = ','.join(sorted(getattr(backend, 'resolved_models', set()))) or backend.model
        except Exception as error:
            response.warnings.append(f'{ranker} ranking unavailable ({type(error).__name__}); returned RRF ranking')
        finally:
            if backend:
                response.usage.update(backend.usage)
                if backend.cost_usd is not None:
                    response.cost_usd = (response.cost_usd or 0) + backend.cost_usd
    response.timings['ranking'] = perf_counter() - start
    if diversity:
        candidates = diverse_order(candidates)
    if anchor_provider is not None:
        candidates = preserve_provider_first(candidates, anchor_provider)
    response.results = budget_results(candidates, max_results, token_budget)
    if assess:
        from .assessment import assessor, summarize
        assessment_backend = None
        start = perf_counter()
        if not response.results:
            response.evidence_assessment = {'status': 'no_results', 'results': [], 'has_potential_evidence': False}
        else:
            try:
                assessment_backend = assessor(client, config, len(response.results))
                judged = await bounded(assessment_backend.rank(question, response.results))
                response.evidence_assessment = summarize(assessment_backend, judged, assessment_threshold)
            except Exception as error:
                response.evidence_assessment = {'status': 'unavailable', 'error_type': type(error).__name__}
                response.warnings.append(f'Evidence assessment unavailable ({type(error).__name__}); usefulness is unknown')
            finally:
                if assessment_backend:
                    response.usage['assessment'] = assessment_backend.usage
                    if assessment_backend.cost_usd is not None:
                        response.cost_usd = (response.cost_usd or 0) + assessment_backend.cost_usd
        response.timings['assessment'] = perf_counter() - start
    response.usage['fast'] = fast
    response.usage['deadline_s'] = deadline
    response.usage['anchor_provider'] = anchor_provider
    response.usage['diversity_enabled'] = diversity
    response.usage['candidate_providers'] = sorted({s.provider for r in candidates for s in r.sources})
    response.usage['output_tokens'] = evidence_tokens(response.results)
    response.usage['candidate_count'] = len(candidates)
    response.timings['total'] = perf_counter() - started
    return response
