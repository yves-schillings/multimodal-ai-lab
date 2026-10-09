"""Check classifier canary execution and one actual scheduled run on local OKD."""
import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import httpx

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.openshift_local_acceptance import PortForward


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--credentials',type=Path,required=True)
    parser.add_argument('--report',type=Path,required=True)
    args=parser.parse_args()
    passwords=json.loads(args.credentials.read_text(encoding='utf-8'))['passwords']
    report={'scope':'Local OKD synthetic classifier canary and MLflow; not production drift monitoring','checks':[],'passed':False}
    def check(name,passed):
        report['checks'].append({'name':name,'passed':bool(passed)})
        print(('PASS ' if passed else 'FAIL ')+name,flush=True)
        if not passed:raise RuntimeError('Monitoring acceptance failed: '+name)
    try:
        with PortForward('multimodal-ai-lab') as forward,httpx.Client(base_url=forward.base,timeout=180,trust_env=False) as client:
            def login(actor):
                response=client.post('/api/auth/login',json={'username':actor,'password':passwords[actor]})
                response.raise_for_status();client.headers['X-Lab-CSRF']=response.json()['csrf']
            login('officer-a')
            check('officer cannot trigger model evaluation',client.post('/api/models/monitoring/run').status_code==403)
            login('reviewer')
            state=client.get('/api/models/monitoring').json()
            check('temporary acceptance interval is sixty seconds',state['enabled'] and state['interval_seconds']==60)
            response=client.post('/api/models/monitoring/run');response.raise_for_status();manual=response.json()
            check('real promoted classifier canary is healthy',manual['status']=='healthy' and manual['metrics']['sample_count']==8)
            report['manual_run']=manual
            before={r['id'] for r in client.get('/api/models/monitoring').json()['runs']}
            scheduled=None
            for _ in range(40):
                time.sleep(2)
                runs=client.get('/api/models/monitoring').json()['runs']
                scheduled=next((r for r in runs if r['id'] not in before),None)
                if scheduled:break
            check('actual recurring evaluation produced a new healthy run',scheduled and scheduled['status']=='healthy' and scheduled['mlflow_run_id']!=manual['mlflow_run_id'])
            report['scheduled_run']=scheduled
            run_id=scheduled['mlflow_run_id']
            if not re.fullmatch('[a-f0-9]{32}',run_id):raise RuntimeError('Unexpected monitoring run identifier.')
            code="import json,os,sys;from mlflow.tracking import MlflowClient;r=MlflowClient(tracking_uri=os.environ['MLFLOW_TRACKING_URI']).get_run(sys.argv[1]);print(json.dumps({'status':r.info.status,'metrics':r.data.metrics,'scope':r.data.tags.get('data_scope')}))"
            result=subprocess.check_output(['oc','exec','-n','multimodal-ai-lab','deployment/lab-app','-c','app','--','python','-c',code,run_id],text=True)
            recorded=json.loads(result)
            check('MLflow server retains completed scheduled metrics',recorded['status']=='FINISHED' and recorded['metrics']['sample_count']==8 and recorded['scope']=='synthetic_canary_only')
            report['mlflow_record']=recorded
        report['passed']=True
    finally:
        args.report.parent.mkdir(parents=True,exist_ok=True)
        args.report.write_text(json.dumps(report,indent=2),encoding='utf-8')


if __name__=='__main__':main()
