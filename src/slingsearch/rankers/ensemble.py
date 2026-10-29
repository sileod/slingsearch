"""Experimental rank-only fusion of model judgments on identical candidates."""
import asyncio
from copy import deepcopy


def fuse_orders(orders, candidates, *, k=60):
    expected = {c.id for c in candidates}
    scores = dict.fromkeys(expected, 0.)
    for order in orders:
        ids = [c.id for c in order]
        if len(ids) != len(expected) or set(ids) != expected:
            raise ValueError('Invalid ensemble candidate mapping')
        for rank, cid in enumerate(ids, 1):
            scores[cid] += 1 / (k + rank)
    copies = deepcopy(candidates)
    for candidate in copies:
        candidate.model_score = scores[candidate.id]
    return sorted(copies, key=lambda c: (-c.model_score, -c.rrf_score, c.id))


class Ensemble:
    def __init__(self, members):
        self.members = members
        self.model = ','.join(member.model for member in members.values())
        self.usage, self.cost_usd, self.resolved_models = {}, None, set()

    async def rank(self, question, candidates):
        self.usage, self.cost_usd, self.resolved_models = {}, None, set()
        names = list(self.members)
        results = await asyncio.gather(*(self.members[name].rank(question, deepcopy(candidates))
                                         for name in names), return_exceptions=True)
        diagnostics = {}
        for name, result in zip(names, results):
            member = self.members[name]
            diagnostics[name] = {'success': not isinstance(result, BaseException),
                                 'model': member.model, 'usage': member.usage,
                                 'reported_cost_usd': member.cost_usd}
            self.resolved_models.update(getattr(member, 'resolved_models', set()) or {member.model})
            if member.cost_usd is not None:
                self.cost_usd = (self.cost_usd or 0) + member.cost_usd
            for key in ('input_tokens', 'output_tokens', 'total_tokens', 'approximate_input_tokens',
                        'ranking_requests', 'decision_questions', 'cached_tokens', 'cache_write_tokens'):
                if key in member.usage:
                    self.usage[key] = self.usage.get(key, 0) + member.usage[key]
        self.usage['ensemble_members'] = diagnostics
        self.usage['ensemble_fusion'] = 'rrf_k60'
        # Fail explicitly rather than silently change the requested ensemble.
        if any(isinstance(result, BaseException) for result in results):
            raise ValueError('One or more ensemble rankers failed')
        return fuse_orders(results, candidates)
