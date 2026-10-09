"""Local CPE inference broker: exact model allowlist, scoped tokens and bounded requests.

No prompt logging, model download, arbitrary URL or cloud fallback. NetworkPolicy
prevents CPEs bypassing this broker. Shared CPU service serializes inference.
"""
import asyncio
import json
import os
import secrets
import time
from collections import defaultdict, deque
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

MODELS = {
    'bge-m3:latest':'7907646426070047a77226ac3e684fbbe8410524f7b4a74d02837e43f2146bab',
    'qwen3:4b':'359d7dd4bcdab3d86b87d73ac27966f4dbb9f5efdfcc75d34a8764a09474fae7'}


def create_gateway(tokens_path=None):
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    token_file = Path(tokens_path or os.environ['LAB_GATEWAY_TOKENS_FILE'])
    semaphore = asyncio.Semaphore(1)
    requests = defaultdict(deque)

    @app.get('/health/ready')
    def ready():
        return {'ready':True}

    @app.api_route('/api/{operation}', methods=['GET','POST'])
    async def inference(operation:str, request:Request):
        tokens = json.loads(token_file.read_text())
        supplied = request.headers.get('authorization','').removeprefix('Bearer ')
        actor = next((key for key,value in tokens.items() if secrets.compare_digest(value,supplied)),None)
        if actor is None:
            raise HTTPException(401,'Scoped local inference credential required')
        if operation not in {'tags','embed','generate'} or request.method != ('GET' if operation=='tags' else 'POST'):
            raise HTTPException(403,'Only approved inference operations are permitted')
        window = requests[actor]
        now = time.monotonic()
        while window and window[0] < now-60:
            window.popleft()
        if len(window) >= 32:
            raise HTTPException(429,'Project inference request budget exceeded')
        window.append(now)
        payload = None
        if operation != 'tags':
            raw = await request.body()
            if len(raw) > 270000:
                raise HTTPException(413,'Inference request exceeds the local bound')
            try:
                payload = json.loads(raw)
                if not isinstance(payload,dict):
                    raise ValueError()
                expected = 'bge-m3:latest' if operation=='embed' else 'qwen3:4b'
                if payload.get('model') != expected:
                    raise HTTPException(403,'Model is not approved for this operation')
                if operation == 'embed':
                    texts=payload.get('input')
                    if not isinstance(texts,list) or not 1<=len(texts)<=128 or any(not isinstance(t,str) or len(t)>2000 for t in texts):
                        raise ValueError()
                    payload={'model':expected,'input':texts,'truncate':False,'keep_alive':0,
                             'options':{'num_thread':2}}
                else:
                    if any(not isinstance(payload.get(k),str) for k in ['prompt','system']) or len(payload['prompt'])>20000 or len(payload['system'])>4000:
                        raise ValueError()
                    payload={'model':expected,'prompt':payload['prompt'],'system':payload['system'],
                             'format':'json','stream':False,'think':False,'keep_alive':0,
                             'options':{'temperature':0,'num_ctx':4096,'num_predict':600,'num_thread':2}}
            except (ValueError,TypeError):
                raise HTTPException(400,'Malformed or unbounded inference request')
        try:
            # Bounded wait avoids unbounded CPU-job queues in this small local cluster.
            await asyncio.wait_for(semaphore.acquire(),timeout=5)
        except TimeoutError:
            raise HTTPException(429,'Local inference capacity busy; retry later')
        try:
            async with httpx.AsyncClient(timeout=180,trust_env=False,follow_redirects=False) as client:
                tags = await client.get('http://lab-inference:11434/api/tags')
                tags.raise_for_status()
                installed=tags.json().get('models',[])
                if any(not any(m.get('name')==name and m.get('digest')==digest and not m.get('remote_host') for m in installed) for name,digest in MODELS.items()):
                    raise HTTPException(503,'Pinned offline models unavailable')
                if operation=='tags':
                    return {'models':[m for m in installed if m.get('name') in MODELS]}
                response=await client.post('http://lab-inference:11434/api/'+operation,json=payload)
                response.raise_for_status()
                return JSONResponse(response.json())
        except (httpx.HTTPError,ValueError):
            raise HTTPException(503,'Local inference unavailable; no remote fallback')
        finally:
            semaphore.release()
    return app
