"""Actual KServe prediction, MLflow training, workbench authentication and persistence."""
import argparse
import contextlib
import json
import socket
import subprocess
import sys
import time
from pathlib import Path
import httpx

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.local_cpe import Controller


@contextlib.contextmanager
def forward(c,service,port):
    with socket.socket() as s:s.bind(('127.0.0.1',0));local=s.getsockname()[1]
    process=subprocess.Popen(['oc','--kubeconfig',c.kubeconfig,'port-forward','service/'+service,'-n','lab-ai-platform',str(local)+':'+str(port),'--address=127.0.0.1'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    try:
        with httpx.Client(base_url='http://127.0.0.1:'+str(local),trust_env=False,timeout=60,follow_redirects=False) as api:
            for _ in range(60):
                if process.poll() is not None:raise RuntimeError('Local platform forward failed')
                try:api.get('/');break
                except httpx.HTTPError:time.sleep(1)
            yield api
    finally:
        process.terminate();process.wait(timeout=10)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--admin-kubeconfig',type=Path,required=True);p.add_argument('--private-dir',type=Path,required=True)
    args=p.parse_args();c=Controller(args.admin_kubeconfig,args.private_dir);checks=[]
    def check(name,value):
        checks.append({'name':name,'passed':bool(value)});print(('PASS ' if value else 'FAIL ')+name,flush=True)
        if not value:raise RuntimeError('Local AI platform acceptance failed: '+name)
    report={'scope':'Free lightweight local CPU components: KServe Standard, JupyterLab, native training Job, existing MLflow. Not the complete Red Hat OpenShift AI product.','checks':checks,'passed':False}
    try:
        ns='lab-ai-platform';config=json.loads((c.private/'workloads.json').read_text());tokens=json.loads((c.private/'workbench-access.json').read_text())
        c.oc('rollout','status','deployment/lab-workbench','-n',ns,'--timeout=240s')
        inference=c.get('inferenceservice','document-classifier',ns)
        check('KServe reports accepted CPU predictor Ready',any(x['type']=='Ready' and x['status']=='True' for x in inference['status']['conditions']))
        with forward(c,'document-classifier-predictor',80) as api:
            payload={'inputs':[{'name':'text','datatype':'BYTES','shape':[1],'data':['Evidence inventory of collected recording exhibits.']}]}
            check('KServe predictor denies missing credential',api.post('/v2/models/document-classifier/infer',json=payload).status_code==401)
            api.headers['Authorization']='Bearer '+tokens['token']
            response=api.post('/v2/models/document-classifier/infer',json=payload);response.raise_for_status();answer=response.json()
            check('real trained model predicts expected category',answer['outputs'][0]['data']==['evidence_inventory'])
            check('served model matches accepted artifact',answer['artifact_sha256']==config['model_sha256'] and answer['model_version']==config['accepted_classifier'])
        job=c.get('job',config['training_job'],ns)
        check('actual native training Job completed',job['status'].get('succeeded')==1)
        pod=next(x['metadata']['name'] for x in json.loads(c.oc('get','pods','-n',ns,'-l','app=lab-workbench','-o','json').stdout)['items'] if x['status']['phase']=='Running')
        # Workbench is the controlled client, not a public ingress.
        read="import json;from pathlib import Path;print(Path('/results/training.json').read_text())"
        # Mount results read-only into this workbench for the actual training evidence handover.
        result=json.loads(c.oc('exec','-n',ns,pod,'--','python','-c',read).stdout)
        report['training']={k:result[k] for k in ['version','accuracy','macro_f1','training_samples','test_samples','mlflow_run_id','artifact_sha256','mlflow_status']}
        check('real classifier training uses separate24/8 split',result['training_samples']==24 and result['test_samples']==8 and result['mlflow_status']=='recorded_on_server')
        # Independent readback from the tracking server through its own pod.
        mlpod=next(x['metadata']['name'] for x in json.loads(c.oc('get','pods','-n','multimodal-ai-lab','-l','app.kubernetes.io/name=lab-mlflow','-o','json').stdout)['items'] if x['status']['phase']=='Running')
        code="import json;from mlflow.tracking import MlflowClient;r=MlflowClient(tracking_uri='http://127.0.0.1:5000').get_run("+repr(result['mlflow_run_id'])+");print(json.dumps({'status':r.info.status,'metrics':r.data.metrics}))"
        logged=json.loads(c.oc('exec','-n','multimodal-ai-lab',mlpod,'--','python','-c',code).stdout)
        check('MLflow independently confirms completed run and metrics',logged['status']=='FINISHED' and logged['metrics']['accuracy']==result['accuracy'] and logged['metrics']['macro_f1']==result['macro_f1'])
        with forward(c,'lab-workbench',8888) as api:
            check('workbench denies unauthenticated contents',api.get('/api/contents').status_code in {302,403})
            api.headers['Authorization']='token '+tokens['jupyter']
            check('workbench authenticated API available',api.get('/api/contents').status_code==200)
            notebook={'cells':[{'cell_type':'markdown','metadata':{},'source':'# Local model training and serving\nThe CPU Job trains TF-IDF + logistic regression on24 synthetic examples and evaluates8 separate examples. Inspect training.json and the local MLflow run. No LLM fine-tuning or production accuracy is claimed.'},
              {'cell_type':'code','metadata':{},'execution_count':None,'outputs':[],
               'source':"import json\nfrom pathlib import Path\ntraining=json.loads(Path('/results/training.json').read_text())\n{k:training[k] for k in ['accuracy','macro_f1','training_samples','test_samples','mlflow_run_id']}"},
              {'cell_type':'code','metadata':{},'execution_count':None,'outputs':[],
               'source':"import urllib.request\nrequest=urllib.request.Request('http://document-classifier-predictor/v2/models/document-classifier/infer',data=json.dumps({'inputs':[{'name':'text','datatype':'BYTES','shape':[1],'data':['Evidence inventory of collected recording exhibits.']}]}).encode(),headers={'Content-Type':'application/json','Authorization':'Bearer '+Path('/credentials/token').read_text().strip()})\njson.loads(urllib.request.urlopen(request).read())"}],
               'metadata':{'kernelspec':{'name':'python3','display_name':'Python3','language':'python'}},'nbformat':4,'nbformat_minor':5}
            saved=api.put('/api/contents/Local-model-training.ipynb',json={'type':'notebook','format':'json','content':notebook});saved.raise_for_status()
            check('native editable training notebook saved',saved.json()['type']=='notebook')
            kernel=api.post('/api/kernels',json={'name':'python3'});kernel.raise_for_status()
            check('actual Python workbench kernel starts',kernel.json()['execution_state'] in {'starting','idle','busy'})
            api.delete('/api/kernels/'+kernel.json()['id']).raise_for_status()
        # Verify a real workbench-origin network/model call, token read inside pod.
        code="import json,urllib.request;from pathlib import Path;r=urllib.request.Request('http://document-classifier-predictor/v2/models/document-classifier/infer',data=json.dumps({'inputs':[{'name':'text','data':['Evidence inventory of collected recording exhibits.']}]}).encode(),headers={'Content-Type':'application/json','Authorization':'Bearer '+Path('/credentials/token').read_text().strip()});print(json.loads(urllib.request.urlopen(r).read())['outputs'][0]['data'][0])"
        check('workbench calls private KServe predictor',c.oc('exec','-n',ns,pod,'--','python','-c',code).stdout.strip()=='evidence_inventory')
        probe="import socket;s=socket.socket();s.settimeout(3);print('OPEN' if s.connect_ex(('1.1.1.1',443))==0 else 'CLOSED')"
        check('workbench has no Internet egress',c.oc('exec','-n',ns,pod,'--','python','-c',probe).stdout.strip()=='CLOSED')
        pvc=c.get('pvc','workbench-data',ns)['metadata']['uid'];c.oc('delete','pod',pod,'-n',ns)
        c.oc('rollout','status','deployment/lab-workbench','-n',ns,'--timeout=240s')
        with forward(c,'lab-workbench',8888) as api:
            api.headers['Authorization']='token '+tokens['jupyter']
            check('saved notebook survives workbench replacement',api.get('/api/contents/Local-model-training.ipynb').status_code==200)
        check('workbench PVC identity retained',pvc==c.get('pvc','workbench-data',ns)['metadata']['uid'])
        report['passed']=True
    finally:c.save('platform-acceptance.json',report)

if __name__=='__main__':main()
