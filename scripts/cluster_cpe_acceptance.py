"""Verify real local CPE controls and workflows before operator activation."""
import argparse
import contextlib
import copy
import json
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.local_cpe import Controller,PROJECTS,digest


@contextlib.contextmanager
def forward(controller,key):
    with socket.socket() as s:s.bind(('127.0.0.1',0));port=s.getsockname()[1]
    process=subprocess.Popen(['oc','--kubeconfig',controller.kubeconfig,'port-forward','service/cpe-app',
                             '-n',controller.namespace(key),str(port)+':8770','--address=127.0.0.1'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    try:
        with httpx.Client(base_url='http://127.0.0.1:'+str(port),timeout=30,trust_env=False) as client:
            for _ in range(60):
                if process.poll() is not None:raise RuntimeError('Local port-forward failed')
                try:
                    if client.get('/api/status').status_code==200:break
                except httpx.HTTPError:pass
                time.sleep(1)
            else:raise RuntimeError('Local CPE did not answer')
            yield client
    finally:
        process.terminate()
        try:process.wait(timeout=10)
        except subprocess.TimeoutExpired:process.kill();process.wait()


def login(client,actor,passwords):
    r=client.post('/api/auth/login',json={'username':actor,'password':passwords[actor]});r.raise_for_status()
    client.headers['X-Lab-CSRF']=r.json()['csrf']


def phase(client,expected):
    for _ in range(150):
        if client.get('/api/status').json().get('cpe',{}).get('phase')==expected:return
        time.sleep(1)
    raise RuntimeError('Mounted CPE activation state did not update')


def verify(controller,key):
    ns=controller.namespace(key);checks=[];passed=False
    def check(name,value):
        checks.append({'name':name,'passed':bool(value)})
        print(('PASS ' if value else 'FAIL ')+name,flush=True)
        if not value:raise RuntimeError('CPE acceptance failed: '+name)
    profile=json.loads(controller.get('configmap','cpe-profile',ns)['data']['profile.json'])
    passwords=json.loads((controller.private/('credentials-'+key+'.json')).read_text())['passwords']
    other='b' if key=='a' else 'a'
    before=controller.snapshot(key)
    try:
        if profile['expires']<=time.time():raise RuntimeError('Synthetic owner approval expired')
        pods=json.loads(controller.oc('get','pods','-n',ns,'-l','app=cpe-app','-o','json').stdout)['items']
        pod=next(p for p in pods if p['status']['phase']=='Running');name=pod['metadata']['name']
        def execute(code):
            r=controller.oc('exec','-n',ns,name,'--','python','-c',code,check=False,timeout=45)
            if r.returncode:raise RuntimeError('Probe execution failed; never count it as denial')
            return r.stdout.strip()
        check('restricted non-root application',int(execute('import os;print(os.getuid())'))!=0)
        check('runtime token is not mounted',execute("from pathlib import Path;print(Path('/var/run/secrets/kubernetes.io/serviceaccount/token').exists())")=='False')
        check('approved immutable app image',pod['spec']['containers'][0]['image']==profile['image'])
        pvc=controller.get('pvc','cpe-data',ns)
        check('separate project storage is Bound',pvc['status']['phase']=='Bound' and pvc['spec']['resources']['requests']['storage']=='1Gi')
        quota=controller.get('resourcequota','cpe-budget',ns)['spec']['hard']
        check('CPU memory storage and GPU admission budgets',quota['limits.cpu']=='1' and quota['limits.memory']=='1Gi' and quota['requests.storage']=='1Gi' and quota['requests.nvidia.com/gpu']=='0')
        for resource,verb,target in [('networkpolicies','update',ns),('configmaps','update',ns),('secrets','get',ns),
                                      ('persistentvolumeclaims','delete',ns),('pods','create',ns),('configmaps','delete','lab-cpe-control')]:
            r=controller.oc('auth','can-i',verb,resource,'-n',target,'--as=system:serviceaccount:'+ns+':cpe-runtime',check=False)
            check('runtime cannot '+verb+' '+resource+' in '+target,r.stdout.strip()=='no')
        probe=copy.deepcopy(controller.get('deployment','cpe-app',ns)['spec']['template'])
        probe.update(apiVersion='v1',kind='Pod');probe['metadata']={'name':'cpe-denial-probe','namespace':ns}
        probe['spec']['containers'][0]['image']='unapproved.invalid/model:latest'
        r=controller.oc('create','--dry-run=server','-f','-',payload=probe,check=False)
        check('admission rejects unapproved workload image',r.returncode!=0 and 'CPE pod contract' in r.stderr)
        probe['spec']['containers'][0]['image']=profile['image']
        probe['spec']['volumes'].append({'name':'forbidden-host','hostPath':{'path':'/etc'}})
        r=controller.oc('create','--dry-run=server','-f','-',payload=probe,check=False)
        check('admission rejects host filesystem mounts',r.returncode!=0 and ('CPE pod contract' in r.stderr or 'hostPath' in r.stderr))
        r=controller.oc('create','--dry-run=server','-f','-',payload={'apiVersion':'v1','kind':'PersistentVolumeClaim',
            'metadata':{'name':'cpe-over-quota','namespace':ns},'spec':{'accessModes':['ReadWriteOnce'],'resources':{'requests':{'storage':'2Gi'}}}},check=False)
        check('admission refuses excess project storage',r.returncode!=0 and 'exceeded quota' in r.stderr)
        own=controller.get('service','cpe-app',ns)['spec']['clusterIP']
        other_ip=controller.get('service','cpe-app',controller.namespace(other))['spec']['clusterIP']
        primary=controller.get('service','lab-app','multimodal-ai-lab')['spec']['clusterIP']
        inference=controller.get('service','lab-inference','multimodal-ai-lab')['spec']['clusterIP']
        for target,ip,port,expected in [('own application',own,8770,'OPEN'),('other project',other_ip,8770,'CLOSED'),
                                        ('primary lab',primary,8770,'CLOSED'),('unapproved model endpoint',inference,11434,'CLOSED'),
                                        ('public Internet','1.1.1.1',443,'CLOSED')]:
            result=execute(f"import socket;s=socket.socket();s.settimeout(3);print('OPEN' if s.connect_ex(('{ip}',{port}))==0 else 'CLOSED');s.close()")
            check('network '+target+' '+expected.lower(),result==expected)
        if profile.get('models'):
            gateway=controller.get('service','lab-inference-gateway','multimodal-ai-lab')['spec']['clusterIP']
            result=execute(f"import socket;s=socket.socket();s.settimeout(3);print('OPEN' if s.connect_ex(('{gateway}',8771))==0 else 'CLOSED');s.close()")
            check('approved inference gateway reachable',result=='OPEN')
            check('gateway rejects missing project token',execute("import httpx;print(httpx.get('http://lab-inference-gateway.multimodal-ai-lab.svc:8771/api/tags',trust_env=False).status_code)")=='401')
            check('gateway denies remote or unapproved model',execute("import httpx,os;from pathlib import Path;print(httpx.post('http://lab-inference-gateway.multimodal-ai-lab.svc:8771/api/generate',headers={'Authorization':'Bearer '+Path(os.environ['LAB_INFERENCE_TOKEN_FILE']).read_text()},json={'model':'remote-cloud','prompt':'fictional','system':'fictional'},trust_env=False).status_code)")=='403')
            for forbidden in ['lab_vectors','cpe_vectors_'+other,'postgres']:
                code="import os,psycopg;from pathlib import Path\ntry:\n db=psycopg.connect(host=os.environ['LAB_VECTOR_HOST'],dbname="+repr(forbidden)+",user=os.environ['LAB_VECTOR_USER'],password=Path(os.environ['LAB_VECTOR_PASSWORD_FILE']).read_text().strip(),connect_timeout=3);db.close();print('OPEN')\nexcept psycopg.OperationalError as e:\n if 'permission denied for database' not in str(e):raise\n print('DENIED')"
                check('database role denied '+forbidden,execute(code)=='DENIED')
        with forward(controller,key) as api:
            api.timeout=300
            controller.gate(key,'provisioned');phase(api,'provisioned')
            login(api,PROJECTS[key],passwords)
            check('tenant business API disabled before activation',api.get('/api/cases').status_code==503)
            controller.gate(key,'verifying');phase(api,'verifying')
            check('verification phase denies tenant business access',api.get('/api/cases').status_code==503)
            login(api,'reviewer',passwords)
            check('reviewer verifier can use gated application',api.get('/api/cases').status_code==200)
            check('unapproved model actions refused',api.post('/api/models/train').status_code==403)
            case=api.post('/api/cases',json={'title':'Synthetic isolated CPE acceptance'});case.raise_for_status();case_id=case.json()['id']
            r=api.post('/api/cases/'+case_id+'/upload',files={'file':('fictional.txt',b'The blue bicycle was left beside the library at nine in the morning.','text/plain')},data={'kind':'document'})
            r.raise_for_status();job_id=r.json()['job_id']
            for _ in range(60):
                job=api.get('/api/jobs/'+job_id).json()
                if job['status'] in {'completed','failed'}:break
                time.sleep(1)
            check('actual isolated document job completed',job['status']=='completed')
            case=api.get('/api/cases/'+case_id).json();doc=case['documents'][0]
            r=api.post('/api/cases/'+case_id+'/documents/'+doc['id']+'/review',json={'text':doc['text'],'expected_revision':case['revision']})
            check('reviewed source persisted',r.status_code==200)
            r=api.post('/api/cases/'+case_id+'/question',json={'question':'Where was the blue bicycle left?'})
            check('isolated reviewed evidence retrieval',r.status_code==200 and not r.json()['abstained'])
            if profile.get('models'):
                answer=r.json()
                check('actual offline RAG cites source text',answer['mode']=='local_generative_rag' and bool(answer.get('claims')) and
                      all(claim['quote'] in answer['sources'][claim['source']-1]['text'] for claim in answer['claims']))
                r=api.post('/api/cases/'+case_id+'/question',json={'question':"What is Neptune's chemical composition?"})
                check('unsupported CPE RAG question abstains',r.status_code==200 and r.json()['abstained'])
                current=api.get('/api/cases/'+case_id).json()
                r=api.post('/api/cases/'+case_id+'/documents/'+doc['id']+'/review',json={'text':'The red bicycle was left beside the museum at ten in the morning.','expected_revision':current['revision']})
                r.raise_for_status()
                r=api.post('/api/cases/'+case_id+'/question',json={'question':'Where was the bicycle left?'})
                check('corrected revision replaces old vector evidence',r.status_code==200 and not r.json()['abstained'] and 'museum' in r.json()['answer'].lower() and 'library' not in str(r.json()['sources']).lower())
            r=api.get('/api/cases/'+case_id,params={'actor':PROJECTS[other]})
            check('another actor cannot override reviewer identity',r.status_code==403)
            other_credentials=json.loads((controller.private/('credentials-'+other+'.json')).read_text())['passwords']
            r=api.post('/api/auth/login',json={'username':PROJECTS[other],'password':other_credentials[PROJECTS[other]]})
            check('other project credentials rejected',r.status_code==401)
            check('verification leaves project inactive for tenants',api.get('/api/status').json()['cpe']['active'] is False)
        check('protected resources unchanged by verification',digest(before)==digest(controller.snapshot(key)))
        passed=True
        return {'scope':'actual local OKD synthetic document CPE; approved offline RAG' if profile.get('models') else 'actual local OKD synthetic document/lexical CPE; no approved model endpoints; admission-only storage quota',
                'passed':True,'checks':checks,'profile_digest':digest(profile),'resource_digest':digest(before),
                'verified_at':time.time(),'case_id':case_id,'pvc_uid':pvc['metadata']['uid'],'namespace':ns}
    finally:
        if not passed:controller.gate(key,'revoked')
        controller.save('verification-progress-'+key+'.json',{'checks':checks,'passed':passed})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project',choices=PROJECTS,required=True)
    parser.add_argument('--admin-kubeconfig',type=Path,required=True)
    parser.add_argument('--private-dir',type=Path,required=True)
    args=parser.parse_args();controller=Controller(args.admin_kubeconfig,args.private_dir)
    report=verify(controller,args.project);controller.save('verified-'+args.project+'.json',report)
    print('All actual CPE verification checks passed. Operator activation remains required.')


if __name__=='__main__':main()
