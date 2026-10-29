class RRF:
    model = None
    usage = {}
    cost_usd = None
    async def rank(self, question, candidates):
        return sorted(candidates, key=lambda r: (-r.rrf_score, r.id))
