"""Deterministic identities, provenance-preserving deduplication and RRF."""
from copy import deepcopy
import re
import unicodedata
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit, unquote


def normalize_doi(value):
    if not value:
        return None
    value = re.sub(r'^(?:https?://(?:dx\.)?doi\.org/|doi:\s*)', '', value.strip(), flags=re.I)
    return unquote(value).lower() or None


def canonical_url(url):
    p = urlsplit(url.strip())
    host = (p.hostname or '').lower().encode('idna').decode()
    port = p.port
    if port and not (p.scheme.lower() == 'https' and port == 443 or p.scheme.lower() == 'http' and port == 80):
        host += f':{port}'
    query = [(k, v) for k, v in parse_qsl(p.query, keep_blank_values=True)
             if not k.lower().startswith('utm_') and k.lower() not in {'fbclid', 'gclid', 'msclkid'}]
    return urlunsplit((p.scheme.lower(), host, p.path or '/', urlencode(sorted(query)), ''))


def title_key(title):
    return ' '.join(re.findall(r'\w+', unicodedata.normalize('NFKC', title).casefold()))


def normalize(result):
    result.doi = normalize_doi(result.doi)
    result.url = canonical_url(result.url)
    if 'doi.org/' in result.url:
        result.doi = normalize_doi(result.url)
    result.identifiers = {k.lower(): str(v).strip() for k, v in result.identifiers.items() if v}
    for key in ('corpusid', 'pmid', 'pmcid'):
        if key in result.identifiers:
            result.identifiers[key] = result.identifiers[key].rstrip('/').rsplit('/', 1)[-1]
    if 'openalex' in result.identifiers:
        result.identifiers['openalex'] = result.identifiers['openalex'].rstrip('/').rsplit('/', 1)[-1].upper()
    arxiv = re.search(r'arxiv\.org/(?:abs|pdf)/([^?#]+)', result.url)
    if arxiv:
        result.identifiers.setdefault('arxiv', arxiv[1].removesuffix('.pdf'))
    if 'arxiv' in result.identifiers:
        result.identifiers['arxiv'] = re.sub(r'v\d+$', '', result.identifiers['arxiv'].removeprefix('arXiv:'))
    # ACL landing pages and PDFs refer to the same stable anthology record.
    # Restrict DOI equivalence to ACL's two known publisher DOI namespaces.
    acl = re.fullmatch(r'/([A-Za-z0-9][A-Za-z0-9.-]*?)(?:\.pdf)?/?', urlsplit(result.url).path)
    if urlsplit(result.url).hostname == 'aclanthology.org' and acl:
        result.identifiers.setdefault('acl', acl[1].casefold())
    acl_doi = re.fullmatch(r'10\.(?:18653|3115)/v1/(.+)', result.doi or '')
    if acl_doi:
        result.identifiers.setdefault('acl', acl_doi[1].casefold())
    if 'acl' in result.identifiers:
        result.identifiers['acl'] = result.identifiers['acl'].casefold()
    if result.doi:
        result.identifiers['doi'] = result.doi
        result.id = 'doi:' + result.doi
        result.url = 'https://doi.org/' + result.doi
    else:
        result.id = result.url
    return result


def equivalent(a, b):
    if a.doi and b.doi and a.doi == b.doi:
        return True
    if a.url == b.url:
        return True
    if any(b.identifiers.get(k) == v for k, v in a.identifiers.items()):
        return True
    if a.kind != b.kind and 'repository' in (a.kind, b.kind):
        return False
    scholarly = (a.doi or a.identifiers or a.abstract) and (b.doi or b.identifiers or b.abstract)
    return bool(scholarly and title_key(a.title) and title_key(a.title) == title_key(b.title)
                and (a.year is None or b.year is None or a.year == b.year))


def merge(a, b):
    for name in ('passages', 'authors', 'sources', 'related_urls', 'code_urls', 'tasks', 'dataset_urls'):
        target = getattr(a, name)
        target.extend(x for x in getattr(b, name) if x not in target)
    if a.url != b.url and b.url not in a.related_urls:
        a.related_urls.append(b.url)
    for name in ('abstract', 'doi', 'year', 'venue', 'citation_count'):
        if getattr(a, name) is None:
            setattr(a, name, getattr(b, name))
    a.identifiers.update({k: v for k, v in b.identifiers.items() if k not in a.identifiers})
    normalize(a)


def fuse(results, weights=None):
    groups = []
    for raw in results:
        item = normalize(deepcopy(raw))
        matches = [g for g in groups if equivalent(g, item)]
        if matches:
            base = matches[0]
            merge(base, item)
            for other in matches[1:]:
                merge(base, other)
                groups.remove(other)
        else:
            groups.append(item)
    # A bridging record can reveal identities of earlier groups.
    changed = True
    while changed:
        changed = False
        for i, a in enumerate(groups):
            for b in groups[i + 1:]:
                if equivalent(a, b):
                    merge(a, b)
                    groups.remove(b)
                    changed = True
                    break
            if changed:
                break
    for item in groups:
        ranks = {}
        for source in item.sources:
            if source.rank < 1:
                raise ValueError('Provider ranks must start at one')
            ranks[source.provider] = min(source.rank, ranks.get(source.provider, source.rank))
        item.rrf_score = sum((weights or {}).get(provider, 1.0) / (60 + rank) for provider, rank in ranks.items())
        item.model_score = None
    return sorted(groups, key=lambda r: (-r.rrf_score, r.id))
