import re
from .output import tokens


def select(question, texts, count=2, max_tokens=180):
    terms = set(re.findall(r'\w{3,}', question.casefold()))
    chunks = []
    for text in texts:
        if not text:
            continue
        for paragraph in re.split(r'\n\s*\n', text):
            words = list(re.finditer(r'\S+', paragraph))
            start = 0
            while start < len(words):
                end = start + 1
                while end < len(words) and tokens(paragraph[words[start].start():words[end].end()]) <= max_tokens:
                    end += 1
                chunk = paragraph[words[start].start():words[end - 1].end()]
                if tokens(chunk) <= max_tokens and chunk not in chunks:
                    chunks.append(chunk)
                if end >= len(words):
                    break
                start = max(start + 1, end - 15)
    def score(text):
        words = set(re.findall(r'\w{3,}', text.casefold()))
        return (len(terms & words), min(len(text), 400))
    return sorted(chunks, key=score, reverse=True)[:count]


def representation(result, limit=400):
    text = result.title + '\n' + '\n'.join(result.passages)
    while tokens(text) > limit:
        text = text[:-1]
    return text


async def extract(client, url):
    import trafilatura
    async with client.stream('GET', url, follow_redirects=True) as response:
        response.raise_for_status()
        if 'text/html' not in response.headers.get('content-type', ''):
            return ''
        content = bytearray()
        async for chunk in response.aiter_bytes():
            content.extend(chunk)
            if len(content) > 2_000_000:
                raise ValueError('Page exceeds extraction size limit')
    return trafilatura.extract(content.decode('utf-8', errors='replace')) or ''


def prepare(result, question, count=2):
    """Select from both abstracts and provider passages, retaining attribution."""
    original = list(result.passages)
    metadata = result.passage_metadata
    texts = [*original, *([result.abstract] if result.abstract else [])]
    result.passages = select(question, texts, count)
    result.passage_metadata = []
    for index, passage in enumerate(result.passages):
        for entry in metadata:
            source = entry.get('index', -1)
            if 0 <= source < len(original) and passage in original[source]:
                mapped = {**entry, 'index':index}
                if mapped not in result.passage_metadata:result.passage_metadata.append(mapped)
