import asyncio
from ..models import Result, Source


class Provider:
    name = ''
    rate_limit_delay = None
    def __init__(self, client, key='', mode='web', year=None):
        self.client, self.key, self.mode, self.year = client, key, mode, year
        self.cost_usd = None

    async def request(self, method, url, **kwargs):
        response = await self.client.request(method, url, **kwargs)
        if response.status_code == 429 and self.rate_limit_delay is not None:
            try:
                delay = float(response.headers.get('Retry-After', self.rate_limit_delay))
            except ValueError:
                delay = self.rate_limit_delay
            if 0 <= delay <= 5:
                await asyncio.sleep(delay)
                response = await self.client.request(method, url, **kwargs)
        response.raise_for_status()
        return response.json()

    def result(self, row, rank):
        row = dict(row)
        row.setdefault('id', row['url'])
        row['sources'] = [Source(self.name, rank)]
        return Result(**row)
