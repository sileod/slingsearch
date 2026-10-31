import sys
import types
from slingsearch.models import SearchResponse


async def test_mcp_provider_override_reaches_search_api(monkeypatch):
    tools={};seen={}
    class Server:
        def __init__(self,*args):pass
        def tool(self):
            def register(function):tools[function.__name__]=function;return function
            return register
        def run(self):pass
    for name in ('mcp','mcp.server','mcp.server.fastmcp'):
        monkeypatch.setitem(sys.modules,name,types.ModuleType(name))
    sys.modules['mcp.server.fastmcp'].FastMCP=Server
    async def search(question,**kwargs):
        seen.update(kwargs)
        return SearchResponse(question,kwargs['mode'])
    from slingsearch import api,mcp
    monkeypatch.setattr(api,'asearch',search)
    mcp.main()
    response=await tools['search']('question',providers=['you','expand'])
    assert seen['providers']==['you','expand']
    assert seen['min_relevance']==.5
    assert response['question']=='question'
