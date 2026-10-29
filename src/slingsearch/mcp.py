"""Optional stdio server; all behavior lives in the Python API."""

def main():
    from mcp.server.fastmcp import FastMCP
    from .api import asearch
    server = FastMCP('Evidence search')

    @server.tool()
    async def search(question: str, mode: str = 'web', ranker: str = 'jev',
                     max_results: int = 5, token_budget: int = 1500, min_relevance: int | None = 3,
                     checks: list[str] | None = None, check_threshold: float = .7) -> dict:
        return (await asearch(question, mode=mode, ranker=ranker,
                              max_results=max_results, token_budget=token_budget, min_relevance=min_relevance,
                              checks=checks, check_threshold=check_threshold)).to_dict()

    server.run()
