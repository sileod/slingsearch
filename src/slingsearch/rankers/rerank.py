import math
import asyncio
from copy import deepcopy
from ..passages import representation
from ..output import tokens


class Voyage:
    def __init__(self, client, key, model, text_budget=400):
        self.client, self.key, self.model, self.text_budget = client, key, model, text_budget
        self.usage, self.cost_usd, self.resolved_models = {}, None, set()

    async def rank(self, question, candidates):
        texts = [representation(c, self.text_budget) for c in candidates]
        for attempt in range(2):
            response = await self.client.post('https://openrouter.ai/api/v1/rerank',
                headers={'Authorization': f'Bearer {self.key}'},
                json={'model': self.model, 'query': question, 'documents': texts, 'top_n': len(texts)})
            if response.status_code != 429 or attempt:
                break
            try:
                delay = float(response.headers.get('Retry-After', 2))
            except ValueError:
                delay = 2
            if not 0 <= delay <= 5:
                break
            await asyncio.sleep(delay)
        response.raise_for_status()
        data = response.json()
        self.usage = data.get('usage') or {}
        self.usage['approximate_input_tokens'] = tokens(question + ''.join(texts))
        self.cost_usd = self.usage.get('cost')
        self.resolved_models = {data.get('model', self.model)}
        rows = data['results']
        if len(rows) != len(candidates) or {r['index'] for r in rows} != set(range(len(candidates))):
            raise ValueError('Incomplete or invalid rerank index mapping')
        result = deepcopy(candidates)
        for row in rows:
            score = float(row['relevance_score'])
            if not math.isfinite(score):
                raise ValueError('Invalid reranker score')
            result[row['index']].model_score = score
        return sorted(result, key=lambda r: (-r.model_score, -r.rrf_score, r.id))
