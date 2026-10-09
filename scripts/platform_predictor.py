"""KServe custom CPU predictor for a hash-verified synthetic classifier snapshot."""
import hashlib
import json
import os
import pickle
import secrets
from pathlib import Path
from fastapi import FastAPI,HTTPException,Request

model_file=Path('/model/classifier.pkl')
metadata=json.loads(Path('/model/metadata.json').read_text())
if hashlib.sha256(model_file.read_bytes()).hexdigest()!=metadata['artifact_sha256']:
    raise RuntimeError('Accepted classifier artifact checksum changed')
model=pickle.loads(model_file.read_bytes())
app=FastAPI(docs_url=None,redoc_url=None,openapi_url=None)

@app.get('/v2/health/ready')
def ready():return {'ready':True}

@app.post('/v2/models/document-classifier/infer')
async def infer(request:Request):
    token=Path('/credentials/token').read_text().strip()
    if not secrets.compare_digest(request.headers.get('authorization',''), 'Bearer '+token):
        raise HTTPException(401,'Local serving credential required')
    raw=await request.body()
    if len(raw)>20000:raise HTTPException(413,'Bounded synthetic inference only')
    try:
        data=json.loads(raw)['inputs'][0]['data']
        if not isinstance(data,list) or not 1<=len(data)<=8 or any(not isinstance(t,str) or len(t)>2000 for t in data):raise ValueError()
    except (KeyError,ValueError,IndexError,TypeError):raise HTTPException(400,'Expected one bounded string input batch')
    return {'model_name':'document-classifier','model_version':metadata['version'],
            'outputs':[{'name':'category','datatype':'BYTES','shape':[len(data)],'data':model.predict(data).tolist()}],
            'artifact_sha256':metadata['artifact_sha256'],'data_scope':'synthetic_only'}
