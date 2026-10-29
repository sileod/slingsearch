"""Small, secret-safe provider health and account probes."""
import asyncio
from time import perf_counter
import httpx
from .config import NAMES
from .providers import PROVIDERS

USAGE_URLS = {
    'tavily': 'https://api.tavily.com/usage',
    'firecrawl': 'https://api.firecrawl.dev/v2/team/credit-usage',
    'openalex': 'https://api.openalex.org/rate-limit',
    'serpapi': 'https://serpapi.com/account.json',
    'github': 'https://api.github.com/rate_limit',
}


def selected(providers):
    names = list(dict.fromkeys(NAMES if providers is None else providers))
    if any(name not in NAMES for name in names):
        raise ValueError('Unknown provider')
    return names


def error_status(response):
    if response.status_code in (403, 429) and response.headers.get('x-ratelimit-remaining') == '0': return 'rate_limited'
    if response.status_code in (401, 403): return 'auth_error'
    if response.status_code == 402: return 'quota_exhausted'
    if response.status_code == 429:
        body = response.text.lower()
        return 'quota_exhausted' if any(s in body for s in ('quota', 'credits exhausted', 'monthly limit', 'daily budget')) else 'rate_limited'
    return 'http_error'


def number_fields(data, fields):
    # Never pass raw provider responses through: account APIs can echo credentials.
    return {key: data[key] for key in fields if isinstance(data.get(key), (int, float)) and not isinstance(data[key], bool)}


async def account(client, config, name):
    key = config.api_keys.get(name, '')
    if not key and name not in ('github', 'papers_with_code', 'huggingface'): return {'provider': name, 'status': 'unconfigured'}
    if name not in USAGE_URLS: return {'provider': name, 'status': 'unsupported', 'remaining': None}
    start = perf_counter()
    try:
        kwargs = {'params': {'api_key': key}} if name == 'serpapi' else {'headers': {'Authorization': f'Bearer {key}'}}
        response = await client.get(USAGE_URLS[name], **kwargs)
        if not response.is_success:
            return {'provider': name, 'status': error_status(response), 'http_status': response.status_code,
                    'latency_s': perf_counter() - start}
        data = response.json()
        values, unit = {}, 'credits'
        if name == 'firecrawl':
            values = number_fields(data.get('data') or {}, ['remainingCredits', 'planCredits'])
        elif name == 'tavily':
            values = {'key': number_fields(data.get('key') or {}, ['usage', 'limit']),
                      'account': number_fields(data.get('account') or {}, ['plan_usage', 'plan_limit', 'paygo_usage', 'paygo_limit'])}
            for section, used, limit in [('key', 'usage', 'limit'), ('account', 'plan_usage', 'plan_limit')]:
                if used in values[section] and limit in values[section]:
                    values[section]['remaining'] = max(0, values[section][limit] - values[section][used])
        elif name == 'serpapi':
            unit = 'searches'
            values = number_fields(data, ['searches_per_month', 'this_month_usage', 'total_searches_left',
                                           'plan_searches_left', 'extra_credits', 'account_rate_limit_per_hour'])
        elif name == 'github':
            unit = 'requests'
            values = {k: number_fields(v, ['limit', 'remaining', 'used', 'reset']) for k, v in data.get('resources', {}).items() if k in ('core', 'search')}
        elif name == 'openalex':
            values = number_fields(data.get('rate_limit') or data, ['credits_limit', 'credits_used', 'credits_remaining'])
        remaining = values.get('remainingCredits', values.get('total_searches_left', values.get('credits_remaining')))
        if name == 'github': remaining = values.get('search', {}).get('remaining')
        if name == 'tavily':
            remaining = min((v['remaining'] for v in values.values() if 'remaining' in v), default=None)
        return {'provider': name, 'status': 'quota_exhausted' if remaining is not None and remaining <= 0 else 'ok',
                'unit': unit, 'remaining': remaining, 'values': values, 'latency_s': perf_counter() - start}
    except Exception as error:
        return {'provider': name, 'status': 'timeout' if isinstance(error, httpx.TimeoutException) else 'error',
                'error_type': type(error).__name__, 'latency_s': perf_counter() - start}


async def usage(client, config, *, providers=None):
    return {'providers': await asyncio.gather(*(account(client, config, name) for name in selected(providers))),
            'note': 'Provider-reported units, not USD or guaranteed free credits; unsupported balances are unknown.'}


async def doctor(client, config, *, providers=None):
    async def probe(name):
        if name in USAGE_URLS:
            row = await account(client, config, name)
            return {**row, 'probe': 'account', 'search_verified': False}
        if not config.api_keys.get(name) and name not in ('papers_with_code', 'huggingface'): return {'provider': name, 'status': 'unconfigured'}
        start = perf_counter()
        try:
            docs = await PROVIDERS[name](client, config.api_keys[name], 'scholar' if name == 'semantic_scholar' else 'web').search('information retrieval', 1)
            return {'provider': name, 'status': 'ok', 'probe': 'search', 'search_verified': True,
                    'results': len(docs), 'latency_s': perf_counter() - start}
        except Exception as error:
            status = error_status(error.response) if isinstance(error, httpx.HTTPStatusError) else 'timeout' if isinstance(error, httpx.TimeoutException) else 'error'
            return {'provider': name, 'status': status, 'error_type': type(error).__name__,
                    'latency_s': perf_counter() - start}
    return {'providers': await asyncio.gather(*(probe(name) for name in selected(providers))),
            'note': 'Account probes verify account access; providers without account endpoints use a one-result search that may consume credits.'}
