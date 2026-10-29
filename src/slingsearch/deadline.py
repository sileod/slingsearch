"""Collect provider tasks with a bounded grace period and explicit cancellation."""
import asyncio
from time import perf_counter
from .fusion import fuse


async def collect(retrieve, names, response, *, stop_at=None, fast=False, min_results=5, grace_s=.15, year=None):
    tasks = {asyncio.create_task(retrieve(name)): name for name in names}
    pending, batches = set(tasks), {}
    grace_end = None
    try:
        while pending:
            end = min(t for t in (stop_at, grace_end) if t is not None) if stop_at is not None or grace_end is not None else None
            timeout = max(0, end - perf_counter()) if end is not None else None
            done, pending = await asyncio.wait(pending, timeout=timeout, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                batches[tasks[task]] = task.result()
            if not done:
                break
            if fast and grace_end is None:
                documents = fuse([r for batch in batches.values() for r in batch])
                usable = [r for r in documents if r.title and r.url and (r.abstract or any(p.strip() for p in r.passages))
                          and (not year or r.year is not None and year[0] <= r.year <= year[1])]
                active = len(names) - len(response.skipped)
                if len(usable) >= min_results and len(response.successful) >= min(2, active):
                    grace_end = perf_counter() + grace_s
        for task in pending:
            name = tasks[task]
            response.cancelled[name] = 'deadline' if stop_at is not None and perf_counter() >= stop_at else 'fast_results_ready'
            task.cancel()
        if pending:
            response.warnings.append('Some providers were cancelled; returned evidence uses completed providers only')
        return [batches.get(name, []) for name in names]
    finally:
        for task in tasks:
            if not task.done(): task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
