import asyncio
import httpx
from .config import Config
from .pipeline import run


def _sync(coroutine_factory):
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coroutine_factory())
    raise RuntimeError('Use await asearch() inside a running event loop')


class Client:
    """Reuse pooled connections within one async context, or one sync context."""
    def __init__(self, config=None, *, transport=None):
        self.config = config or Config()
        self.transport = transport
        self._http = None
        self._runner = None

    async def __aenter__(self):
        if self._http is not None:
            raise RuntimeError('Client is already open')
        self._http = httpx.AsyncClient(timeout=self.config.timeout, transport=self.transport,
                                      limits=httpx.Limits(max_connections=20))
        return self

    async def __aexit__(self, *args):
        await self._http.aclose()
        self._http = None

    async def asearch(self, question, **kwargs):
        if self._http is None:
            async with self:
                return await run(self._http, self.config, question, **kwargs)
        return await run(self._http, self.config, question, **kwargs)

    async def _operation(self, operation, **kwargs):
        if self._http is None:
            async with self:
                return await operation(self._http, self.config, **kwargs)
        return await operation(self._http, self.config, **kwargs)

    def _sync_operation(self, operation, **kwargs):
        factory = lambda: self._operation(operation, **kwargs)
        if self._runner is not None:
            return self._runner.run(factory())
        if self._http is not None:
            raise RuntimeError('Use the async method with an async client')
        return _sync(factory)

    async def ausage(self, *, providers=None):
        from .operations import usage
        return await self._operation(usage, providers=providers)

    def usage(self, *, providers=None):
        from .operations import usage
        return self._sync_operation(usage, providers=providers)

    async def adoctor(self, *, providers=None):
        from .operations import doctor
        return await self._operation(doctor, providers=providers)

    def doctor(self, *, providers=None):
        from .operations import doctor
        return self._sync_operation(doctor, providers=providers)

    async def acitation_graph(self, paper_id, *, direction='both', limit=10):
        from .graph import citation_graph
        return await self._operation(citation_graph, paper_id=paper_id, direction=direction, limit=limit)

    def citation_graph(self, paper_id, *, direction='both', limit=10):
        from .graph import citation_graph
        return self._sync_operation(citation_graph, paper_id=paper_id, direction=direction, limit=limit)

    async def anovelty_search(self, contribution, *, related_work=(), **kwargs):
        from .review import review
        return await self._operation(review, workflow='novelty', contribution=contribution, papers=related_work, **kwargs)

    def novelty_search(self, contribution, *, related_work=(), **kwargs):
        from .review import review
        return self._sync_operation(review, workflow='novelty', contribution=contribution, papers=related_work, **kwargs)

    async def arelated_work_search(self, papers, *, contribution=None, **kwargs):
        from .review import review
        return await self._operation(review, workflow='related_work', contribution=contribution, papers=papers, **kwargs)

    def related_work_search(self, papers, *, contribution=None, **kwargs):
        from .review import review
        return self._sync_operation(review, workflow='related_work', contribution=contribution, papers=papers, **kwargs)

    def __enter__(self):
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            raise RuntimeError('Use async with Client() inside a running event loop')
        self._runner = asyncio.Runner()
        self._runner.run(self.__aenter__())
        return self

    def __exit__(self, *args):
        try:
            self._runner.run(self.__aexit__(*args))
        finally:
            self._runner.close()
            self._runner = None

    def search(self, question, **kwargs):
        if self._runner is not None:
            return self._runner.run(self.asearch(question, **kwargs))
        if self._http is not None:
            raise RuntimeError('Use asearch with an async client')
        return _sync(lambda: self.asearch(question, **kwargs))


async def asearch(question, *, config=None, **kwargs):
    async with Client(config) as client:
        return await client.asearch(question, **kwargs)


def search(question, *, config=None, **kwargs):
    return _sync(lambda: asearch(question, config=config, **kwargs))


async def anovelty_search(contribution, *, config=None, **kwargs):
    async with Client(config) as client:
        return await client.anovelty_search(contribution, **kwargs)


def novelty_search(contribution, *, config=None, **kwargs):
    return _sync(lambda: anovelty_search(contribution, config=config, **kwargs))


async def arelated_work_search(papers, *, config=None, **kwargs):
    async with Client(config) as client:
        return await client.arelated_work_search(papers, **kwargs)


def related_work_search(papers, *, config=None, **kwargs):
    return _sync(lambda: arelated_work_search(papers, config=config, **kwargs))
