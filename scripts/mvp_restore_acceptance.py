"""Serve a private recovery COPY on loopback; never replace the running lab's data."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import sqlite3
import subprocess
import sys
import time
import uuid

import httpx

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from lab.backend_store import LabStore
from scripts.deploy_local_auth import protect_file


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backup-dir',type=Path,required=True,help='Trusted restored-application directory with manifest.json')
    parser.add_argument('--credential-config',type=Path,required=True,help='Private hashed local account configuration')
    parser.add_argument('--credentials',type=Path,required=True,help='Private acceptance login passwords')
    parser.add_argument('--private-dir',type=Path,required=True)
    args=parser.parse_args()
    paths=[p.resolve() for p in (args.backup_dir,args.credential_config,args.credentials,args.private_dir)]
    if any(p==ROOT or ROOT in p.parents for p in paths):
        raise ValueError('Recovery data and credentials must remain outside the code checkout.')
    backup,credential_config,credentials,target=paths
    target.mkdir(parents=True,exist_ok=True)
    run=target/('serving-'+uuid.uuid4().hex[:12]);run.mkdir()
    report={'passed':False,'scope':'Isolated host loopback recovery: one saved case, local authentication and existing classifier. No restored RAG, MLflow server or cluster claim.','started_at':time.time(),'checks':[]}
    def check(name,value):
        report['checks'].append({'name':name,'passed':bool(value)})
        print(('PASS ' if value else 'FAIL ')+name,flush=True)
        if not value:raise RuntimeError('Recovery acceptance failed: '+name)
    process=None
    try:
        manifest=json.loads((backup/'manifest.json').read_text())
        for name,expected in manifest.items():
            source=(backup/name).resolve()
            if backup not in source.parents or not source.is_file() or source.is_symlink():
                raise ValueError('Invalid recovery manifest path.')
            if hashlib.sha256(source.read_bytes()).hexdigest()!=expected:
                raise ValueError('Recovery manifest checksum mismatch.')
        restored=run/'application';shutil.copytree(backup,restored)
        for p in restored.rglob('*'):
            if p.is_file():protect_file(p)
        check('restored application files match trusted backup hashes',all(hashlib.sha256((restored/n).read_bytes()).hexdigest()==v for n,v in manifest.items()))
        for name in ('statements.sqlite3','identity.sqlite3'):
            with sqlite3.connect(restored/name) as db:
                check(name+' integrity',db.execute('PRAGMA integrity_check').fetchone()[0]=='ok')
                if name=='statements.sqlite3':
                    check('no pending jobs can replay source paths',db.execute("SELECT count(*) FROM jobs WHERE status IN ('pending','running')").fetchone()[0]==0)
                    case_id=db.execute("SELECT id FROM cases WHERE owner_actor='officer-a' ORDER BY updated_at DESC LIMIT 1").fetchone()[0]
                    state=json.loads(db.execute("SELECT value FROM settings WHERE key='model_state'").fetchone()[0])
                    version=state['active_version']
                    metadata=json.loads(db.execute('SELECT metadata_json FROM model_versions WHERE version=?',(version,)).fetchone()[0])
        artifact=restored/'models'/(version+'.pkl')
        check('active classifier matches registered artifact checksum',hashlib.sha256(artifact.read_bytes()).hexdigest()==metadata['artifact_sha256'])
        expected_store=LabStore(restored)
        expected_store.simulated_identity=False
        expected_case=expected_store.get_case('officer-a',case_id)
        env={k:v for k,v in os.environ.items() if not k.startswith(('LAB_','MLFLOW_'))}
        env.update(LAB_DATA_DIR=str(restored),LAB_AUTH_MODE='local_sessions',LAB_CREDENTIAL_FILE=str(credential_config),LAB_MONITORING='disabled',LAB_RETRIEVAL_BACKEND='lexical',LAB_NETWORK_MODE='loopback')
        with socket.socket() as sock:
            sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
        passwords=json.loads(credentials.read_text())['passwords']
        with (run/'server-private.log').open('w',encoding='utf-8') as log:
            process=subprocess.Popen([sys.executable,'-m','uvicorn','app:app','--host','127.0.0.1','--port',str(port)],cwd=ROOT,env=env,stdout=log,stderr=log)
            with httpx.Client(base_url='http://127.0.0.1:'+str(port),timeout=30,trust_env=False) as api:
                for _ in range(60):
                    if process.poll() is not None:raise RuntimeError('Recovery server failed; inspect the private server log.')
                    try:
                        if api.get('/health/ready').status_code==200:break
                    except httpx.HTTPError:pass
                    time.sleep(1)
                else:raise RuntimeError('Recovery server did not become ready.')
                check('actual restored HTTP server ready',api.get('/health/ready').status_code==200)
                check('restored case inaccessible without login',api.get('/api/cases/'+case_id).status_code==401)
                def login(actor):
                    response=api.post('/api/auth/login',json={'username':actor,'password':passwords[actor]});response.raise_for_status()
                    api.headers['X-Lab-CSRF']=response.json()['csrf']
                login('officer-a')
                response=api.get('/api/cases/'+case_id);response.raise_for_status()
                check('authenticated restored case contents equal backup',response.json()==expected_case)
                login('officer-b')
                check('other owner denied restored case',api.get('/api/cases/'+case_id).status_code==403)
                login('reviewer')
                response=api.post('/api/models/predict',json={'text':'Evidence inventory of collected recording exhibits.'});response.raise_for_status()
                prediction=response.json()
                check('restored classifier serves actual prediction',prediction['status']=='classified' and prediction['label']=='evidence_inventory' and prediction['model_version']==version)
                check('restored server uses real local authentication',api.get('/api/status').json()['simulated_identity'] is False)
        report.update(passed=True,case_id=case_id,classifier_version=version,artifact_sha256=metadata['artifact_sha256'])
    finally:
        if process is not None:
            process.terminate()
            try:process.wait(timeout=15)
            except subprocess.TimeoutExpired:process.kill();process.wait()
        report['finished_at']=time.time()
        path=run/'acceptance.json';path.write_text(json.dumps(report,indent=2));protect_file(path)
        if (run/'server-private.log').exists():protect_file(run/'server-private.log')
        print('Private restored-serving record: '+str(run))


if __name__=='__main__':main()
