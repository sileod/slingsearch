"""Tiny unjudged discovery comparison. Uses shared cache/budget, no models."""
import asyncio
from dataclasses import asdict
import json
from pathlib import Path
from time import perf_counter
import httpx
from slingsearch import Config
from slingsearch.fusion import fuse
from slingsearch.providers import PROVIDERS
from benchmarks.decision_budget import Transport


async def run():
    config=Config();names=('exa','expand','you');rows=[];raw=[]
    async with httpx.AsyncClient(transport=Transport('benchmarks/reports/decision-audit'),timeout=35) as client:
        for i,question in enumerate(['What are the limitations of approximate nearest neighbor search?',
                                     'How do retrieval systems generalize across domains?']):
            docs={};status={}
            for name in names:
                start=perf_counter()
                try:
                    docs[name]=fuse(await PROVIDERS[name](client,config.api_keys.get(name,''),'web').search(question,5))
                    status[name]={'status':'complete','count':len(docs[name]),'text_bearing':sum(bool(r.passages) for r in docs[name])}
                except Exception as error:
                    docs[name]=[];status[name]={'status':'failed','error_type':type(error).__name__}
                status[name]['seconds']=perf_counter()-start
            urls={name:{r.url for r in found} for name,found in docs.items()}
            for name in names:
                status[name]['urls_absent_from_other_two']=len(urls[name]-set().union(*(urls[n] for n in names if n!=name)))
            rows.append({'query_id':i,'providers':status})
            raw.append({'query_id':i,'question':question,'results':{n:[asdict(r) for r in d] for n,d in docs.items()}})
    root=Path('benchmarks/reports/new-provider-pilot');root.mkdir(exist_ok=True,parents=True)
    (root/'raw.private.json').write_text(json.dumps(raw))
    (root/'summary.json').write_text(json.dumps({'queries':2,'rows':rows,
        'limitations':['No relevance judgments; URL diversity is not evidence quality',
            'Different URLs can be copies or versions of the same document; this is not document-level novelty',
            'Two hand-picked questions, five results per provider, not an independent benchmark',
            'No model ranking or answer generation; exact responses cached privately']},indent=2))
    lines=['# Expand and You.com discovery smoke comparison','',
        '| Provider | Successful queries | Results | Text-bearing results | URLs absent from the other two (summed per query) |',
        '|---|---:|---:|---:|---:|']
    for name in names:
        values=[r['providers'][name] for r in rows]
        lines.append(f"| {name} | {sum(v['status']=='complete' for v in values)}/2 | {sum(v.get('count',0) for v in values)} | {sum(v.get('text_bearing',0) for v in values)} | {sum(v['urls_absent_from_other_two'] for v in values)} |")
    lines+=['','Unjudged URL diversity is not a quality or relevance gain; copies or paper versions can have different URLs. Two hand-picked questions do not justify changing defaults.',
        'Reproduce: `python -m benchmarks.new_provider_pilot` with keys configured. Uses the shared experiment budget/cache. No models or full-page extraction are called.']
    (root/'README.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))


if __name__=='__main__':asyncio.run(run())
