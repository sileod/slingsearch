"""Private experiment transport: persistent cross-process budget and exact replay.

No credentials, question text, or source text are written to the budget ledger.
Provider costs not reported in USD retain explicit conservative allowances.
"""
import fcntl
import hashlib
import json
from pathlib import Path
import uuid
import httpx


class Budget:
    def __init__(self,root):
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True)
        self.path=self.root/'api-budget.json'
    def edit(self,callback):
        with (self.root/'budget.lock').open('a+') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            data=json.loads(self.path.read_text()) if self.path.exists() else {
                'prior_session_estimated_usd':1.890692856,'new_limit_usd':2.5,
                'uncertainty_allowance_usd':.10,'charged_usd':0.,'outstanding':{},
                'calls':{},'cache_hits':0,'scope':'New decision audit only; unreported provider charges use conservative per-call allowances'}
            value=callback(data)
            pending=self.path.with_suffix('.next.json')
            pending.write_text(json.dumps(data,indent=2))
            pending.replace(self.path)
            return value
    def reserve(self,group,amount):
        token=uuid.uuid4().hex
        def change(d):
            reserved=d['charged_usd']+sum(v['allowance_usd'] for v in d['outstanding'].values())+amount
            if reserved>d['new_limit_usd'] or d['prior_session_estimated_usd']+reserved+d['uncertainty_allowance_usd']>5:
                raise ValueError('Global experiment budget reservation stop')
            d['outstanding'][token]={'group':group,'allowance_usd':amount}
            d['calls'][group]=d['calls'].get(group,0)+1
        self.edit(change);return token
    def settle(self,token,cost):
        def change(d):
            allowance=d['outstanding'].pop(token)['allowance_usd']
            actual=allowance if cost is None else cost
            if actual>allowance:raise ValueError('Reported cost exceeds request allowance')
            d['charged_usd']+=actual
        self.edit(change)


class Transport(httpx.AsyncBaseTransport):
    def __init__(self,root):
        self.budget=Budget(root);self.inner=httpx.AsyncHTTPTransport(retries=0)
        self.cache=Path(root)/'cache';self.cache.mkdir(exist_ok=True)
    async def handle_async_request(self,request):
        host,path=request.url.host,request.url.path
        payload=json.loads(request.content) if request.content else {}
        group=host
        if host=='openrouter.ai':
            if path=='/api/alpha/decisions' and payload.get('model') in ['typesafe/jev-1.14','~typesafe/jev-latest','perplexity/pplx-decider-v1.2','liquid/d1-ultra','~liquid/d1-ultra-latest']:
                group=payload['model'];amount=max(.0002,len(request.content)*.000000042*3)
            elif path=='/api/v1/rerank' and payload.get('model')=='voyageai/rerank-3-lite':
                group='voyage';amount=max(.0001,len(request.content)*.00000002*3)
            elif path=='/api/v1/chat/completions' and payload.get('model')=='openai/gpt-4o-2024-11-20' and payload.get('max_tokens')==512:
                import tiktoken
                n=sum(len(tiktoken.get_encoding('o200k_base').encode(m['content'])) for m in payload['messages'])+200
                group='independent_gpt4o_judge';amount=n*2.5/1e6+512*10/1e6
            else:raise ValueError('OpenRouter request outside experiment scope')
        else:
            allowances={'api.exa.ai':.05,'api.parallel.ai':.01,'api.search.brave.com':.01,
                        'api.tavily.com':.02,'api.firecrawl.dev':.05,'api.openalex.org':.01,
                        'api.semanticscholar.org':0,'serpapi.com':.01,
                        'api.github.com':0,'paperswithcode.co':0,'huggingface.co':0}
            if host not in allowances:raise ValueError('Provider outside experiment scope')
            amount=allowances[host]
        digest=hashlib.sha256(str(request.url).encode()+request.content).hexdigest()
        cached=self.cache/(digest+'.json')
        if cached.exists():
            def hit(d):d['cache_hits']+=1
            self.budget.edit(hit)
            saved=json.loads(cached.read_text());return httpx.Response(200,json=saved)
        token=self.budget.reserve(group,amount)
        response=await self.inner.handle_async_request(request);await response.aread()
        if not response.is_success:
            def failed(d):
                errors=d.setdefault('http_errors',{})
                label=group+':'+str(response.status_code)
                errors[label]=errors.get(label,0)+1
            self.budget.edit(failed)
        data=response.json() if response.is_success else {}
        cost=None
        if host=='openrouter.ai':cost=(data.get('usage') or {}).get('cost')
        elif host=='api.exa.ai':cost=(data.get('costDollars') or {}).get('total')
        elif host=='api.openalex.org':cost=(data.get('meta') or {}).get('cost_usd')
        elif host=='api.parallel.ai':cost=.005
        elif host=='api.semanticscholar.org':cost=0
        # A rejected request has no successful search/model execution.
        if not response.is_success:cost=0
        self.budget.settle(token,cost)
        if response.is_success:
            secrets = [request.url.params.get('api_key', '')]
            for name in ('authorization', 'x-api-key', 'x-subscription-token'):
                value = request.headers.get(name, '')
                secrets.append(value.removeprefix('Bearer '))
            serialized = json.dumps(data)
            for secret in secrets:
                if len(secret) >= 8:
                    serialized = serialized.replace(secret, '[REDACTED]')
            pending=cached.with_suffix('.next.json');pending.write_text(serialized);pending.replace(cached)
        return response
    async def aclose(self):await self.inner.aclose()
