"""Loopback-only synthetic Secure Case AI Lab. Actor selection is not authentication."""
import hashlib
import importlib.util
import os
import shutil
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import urlsplit

from fastapi import FastAPI, File, Form, UploadFile, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, StrictInt

from lab.backend_store import LabStore
from lab.jobs import JobRunner
from lab.models import ModelRegistry
from lab.retrieval import answer_question
from lab.speech import SpeechEngine
from lab.store import ConflictError
from lab.deployment_policy import deployment_plan

BASE=Path(__file__).resolve().parent
MAX_UPLOAD=25*1024*1024
LOOPBACK={'127.0.0.1','localhost','::1'}


class CaseInput(BaseModel):
    title: str=Field(min_length=1,max_length=200)


class ReviewInput(BaseModel):
    text: str | None=None
    expected_revision: StrictInt


class QuestionInput(BaseModel):
    question: str=Field(min_length=1,max_length=2000)


class WorkflowInput(BaseModel):
    question: str | None=Field(default=None,max_length=2000)


class PromoteInput(BaseModel):
    version: str | None=None


class PredictInput(BaseModel):
    text: str=Field(min_length=1,max_length=300000)


def create_app(data_dir=None):
    store=LabStore(Path(data_dir or os.environ.get('LAB_DATA_DIR',BASE/'data')))
    registry=ModelRegistry(store)
    speech=SpeechEngine(model_dir=Path(os.environ.get('LAB_WHISPER_MODEL_DIR',BASE/'models'/'whisper-base')))
    runner=JobRunner(store,registry,speech)

    @asynccontextmanager
    async def lifespan(application):
        runner.recover()
        yield
        runner.close()

    app=FastAPI(title='AI Lab: Multimodal Case Processing',lifespan=lifespan)

    @app.get('/api/deployment-plan')
    def planned_deployment():
        return deployment_plan()
    app.state.store,app.state.registry,app.state.runner=store,registry,runner

    @app.middleware('http')
    async def local_only(request:Request,call_next):
        host=request.url.hostname
        # Starlette's in-process TestClient is the only testserver exception.
        test_client=request.client and request.client.host=='testclient' and host=='testserver'
        if host not in LOOPBACK and not test_client:
            return JSONResponse({'detail':'This synthetic demonstration only accepts loopback hosts.'},status_code=403)
        if request.client and request.client.host not in LOOPBACK and not test_client:
            return JSONResponse({'detail':'Remote clients are disabled in synthetic identity mode.'},status_code=403)
        origin=request.headers.get('origin')
        if origin:
            parsed=urlsplit(origin)
            # Require the exact host and port, preventing a foreign local webpage
            # from using the actor selector as an unauthenticated mutation API.
            if parsed.scheme not in {'http','https'} or parsed.netloc != request.headers.get('host'):
                return JSONResponse({'detail':'Cross-origin access is disabled.'},status_code=403)
        # Authorize before FastAPI parses multipart or JSON request bodies.
        if request.url.path.startswith('/api/') and request.url.path not in {'/api/status','/api/actors'}:
            selected=request.query_params.get('actor')
            if selected is None:
                return JSONResponse({'detail':'Select a simulated actor.'},status_code=422)
            try:
                store._actor(selected)
                parts=request.url.path.strip('/').split('/')
                if len(parts)>=3 and parts[:2]==['api','cases']:
                    store.get_case(selected,parts[2])
            except PermissionError as exc:
                return JSONResponse({'detail':str(exc)},status_code=403)
            except KeyError:
                return JSONResponse({'detail':'Requested item was not found.'},status_code=404)
        length=request.headers.get('content-length')
        if length:
            try:
                if int(length)>MAX_UPLOAD+1024*1024:
                    return JSONResponse({'detail':'Request exceeds the local upload limit.'},status_code=413)
            except ValueError:
                return JSONResponse({'detail':'Invalid content length.'},status_code=400)
        response=await call_next(request)
        response.headers['X-Content-Type-Options']='nosniff'
        response.headers['Cache-Control']='no-store'
        response.headers['Content-Security-Policy']="default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; connect-src 'self'; img-src 'self' data:; frame-ancestors 'none'"
        return response

    @app.exception_handler(PermissionError)
    async def denied(request,exc):
        return JSONResponse({'detail':str(exc)},status_code=403)

    @app.exception_handler(KeyError)
    async def missing(request,exc):
        return JSONResponse({'detail':'Requested item was not found.'},status_code=404)

    @app.exception_handler(ConflictError)
    async def conflict(request,exc):
        return JSONResponse({'detail':str(exc)},status_code=409)

    @app.exception_handler(ValueError)
    async def invalid(request,exc):
        return JSONResponse({'detail':str(exc)},status_code=422)

    def actor_check(actor):
        return store._actor(actor)

    @app.get('/health/live')
    def live():
        return {'status':'alive'}

    @app.get('/health/ready')
    def ready():
        with store._connection() as db:
            db.execute('SELECT 1').fetchone()
        return {'status':'ready','mode':'synthetic_loopback'}

    @app.get('/api/status')
    def status():
        tesseract=shutil.which('tesseract') is not None
        whisper_ready=speech.model_dir.is_dir() and (speech.model_dir/'model.bin').is_file()
        return {'name':'Multimodal AI Lab','identity_mode':'simulated_loopback_only','simulated_identity':True,
                'production_ready':False,'data_notice':'Synthetic demonstration only. Caller-selected actors are not real authentication.',
                'capabilities':{'document_extraction':True,'classification':registry.state()['active_version'] is not None,
                                'model_training':importlib.util.find_spec('sklearn') is not None,'speech_recognition':whisper_ready and importlib.util.find_spec('faster_whisper') is not None,
                                'ocr':tesseract and importlib.util.find_spec('pytesseract') is not None,'mlflow':importlib.util.find_spec('mlflow') is not None,
                                'retrieval':'case-scoped TF-IDF/lexical extractive','generative_llm':False,
                                'queue':'SQLite durable local single worker; pending jobs recover on restart; at-least-once processing',
                                'rabbitmq':False,'public_cloud_api':False},
                'limits':{'max_upload_bytes':MAX_UPLOAD,'max_audio_seconds':300,'max_pdf_pages':100},
                'speech_model':speech.model_name,'speech_languages':['en','fr','nl']}

    @app.get('/api/actors')
    def actors():
        return {'actors':store.actors()}

    @app.get('/api/cases')
    def cases(actor:str):
        return {'cases':store.list_cases(actor)}

    @app.post('/api/cases')
    def create_case(body:CaseInput,actor:str):
        return store.create_case(actor,body.title)

    @app.get('/api/cases/{case_id}')
    def get_case(case_id:str,actor:str):
        return store.get_case(actor,case_id)

    @app.patch('/api/cases/{case_id}/segments/{segment_id}')
    def review_segment(case_id:str,segment_id:str,body:ReviewInput,actor:str):
        return store.update_segment(actor,case_id,segment_id,body.text,body.expected_revision)

    @app.post('/api/cases/{case_id}/draft')
    def draft(case_id:str,actor:str):
        return store.make_draft(actor,case_id)

    @app.post('/api/cases/{case_id}/approve')
    def approve(case_id:str,body:ReviewInput,actor:str):
        return store.approve(actor,case_id,body.expected_revision)

    @app.post('/api/cases/{case_id}/question')
    def question(case_id:str,body:QuestionInput,actor:str):
        return answer_question(store,actor,case_id,body.question)

    @app.post('/api/cases/{case_id}/workflow')
    def workflow(case_id:str,body:WorkflowInput,actor:str):
        case=store.get_case(actor,case_id)
        steps=[{'step':'authorize_case','status':'completed'}, {'step':'inspect_review_state','status':'completed'}]
        sources=case['segments']+case['documents']
        missing=sum(not s['reviewed'] for s in sources)
        if not sources or missing:
            steps.append({'step':'prepare_draft','status':'blocked','reason':'Human review of every source is required.','unreviewed_sources':missing})
        elif case['draft']:
            steps.append({'step':'prepare_draft','status':'already_available'})
        else:
            case=store.make_draft(actor,case_id)
            steps.append({'step':'prepare_draft','status':'completed'})
        result={'mode':'bounded_deterministic_workflow','steps':steps,'case':case,'approval':'Human reviewer action required; workflow cannot approve.','max_steps':4}
        if body.question:
            result['retrieval']=answer_question(store,actor,case_id,body.question)
            steps.append({'step':'retrieve_reviewed_sources','status':'completed'})
        return result

    @app.post('/api/cases/{case_id}/upload')
    async def upload(case_id:str,actor:str,file:UploadFile=File(...),kind:str=Form('document'),language:str=Form('auto')):
        store.get_case(actor,case_id)  # Before filename, file contents or job access.
        if kind not in {'audio','document'} or language not in {'en','fr','nl','auto'}:
            raise ValueError('Choose document/audio and en/fr/nl/auto.')
        name=(file.filename or 'upload').replace('\\','/').rsplit('/',1)[-1][:150]
        suffix=Path(name).suffix.lower()
        if kind=='document' and suffix not in {'.txt','.md','.csv','.pdf','.png','.jpg','.jpeg','.tif','.tiff'}:
            raise ValueError('Unsupported document file type.')
        uploads=store.root/'uploads'; uploads.mkdir(exist_ok=True)
        path=uploads/(uuid.uuid4().hex+suffix)
        digest=hashlib.sha256((actor+'|'+case_id+'|'+kind+'|'+language).encode())
        size=0
        try:
            with path.open('wb') as target:
                while chunk:=await file.read(65536):
                    size+=len(chunk)
                    if size>MAX_UPLOAD:
                        raise ValueError('Upload exceeds 25 MiB.')
                    digest.update(chunk); target.write(chunk)
            if not size:
                raise ValueError('Upload is empty.')
            job=store.create_job(actor,case_id,kind,{'path':str(path),'name':name,'language':language},digest.hexdigest())
            with store._connection() as db:
                existing=json_load(db.execute('SELECT payload_json FROM jobs WHERE id=?',(job['id'],)).fetchone()['payload_json'])
            if existing.get('path')!=str(path):
                path.unlink(missing_ok=True)
            runner.submit(job['id'])
            return {'job_id':job['id'],'status':job['status']}
        except Exception:
            path.unlink(missing_ok=True)
            raise
        finally:
            await file.close()

    @app.get('/api/jobs/{job_id}')
    def job(job_id:str,actor:str):
        return store.get_job(actor,job_id)

    @app.post('/api/cases/{case_id}/documents/{document_id}/review')
    def review_document(case_id:str,document_id:str,body:ReviewInput,actor:str):
        return store.review_document(actor,case_id,document_id,body.text,body.expected_revision)

    @app.get('/api/models')
    def models(actor:str):
        actor_check(actor)
        return registry.state()

    @app.get('/api/models/runs')
    def runs(actor:str):
        actor_check(actor)
        return {'runs':registry.state()['versions']}

    @app.post('/api/models/train')
    def train(actor:str):
        actor_check(actor)
        job=store.create_job(actor,None,'train',{},uuid.uuid4().hex)
        runner.submit(job['id'])
        return {'job_id':job['id'],'status':job['status']}

    @app.post('/api/models/promote')
    def promote(body:PromoteInput,actor:str):
        return registry.promote(actor,body.version)

    @app.post('/api/models/rollback')
    def rollback(actor:str):
        return registry.rollback(actor)

    @app.post('/api/models/predict')
    def predict(body:PredictInput,actor:str):
        actor_check(actor)
        return registry.classify(body.text)

    if (BASE/'static').is_dir():
        app.mount('/static',StaticFiles(directory=BASE/'static'),name='static')

    @app.get('/')
    def index():
        if (BASE/'static'/'index.html').is_file():
            return FileResponse(BASE/'static'/'index.html')
        return {'name':'Multimodal AI Lab','api_docs':'/docs'}

    @app.get('/{asset}')
    def web_asset(asset:str):
        if asset in {'style.css','app.js','core.js'} and (BASE/'static'/asset).is_file():
            return FileResponse(BASE/'static'/asset)
        return JSONResponse({'detail':'Not found.'},status_code=404)

    return app


def json_load(value):
    import json
    return json.loads(value)


app=create_app()
