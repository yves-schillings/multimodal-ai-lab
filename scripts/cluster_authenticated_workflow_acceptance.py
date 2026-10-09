"""Actual local speech/OCR/classifier integration with credential-bound identities."""
import argparse
import io
import json
import sys
import time
from pathlib import Path

import httpx
from PIL import Image,ImageDraw,ImageFont

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.openshift_local_acceptance import PortForward
from scripts.deploy_local_auth import protect_file


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--credentials',type=Path,required=True)
    parser.add_argument('--fictional-audio',type=Path,required=True)
    parser.add_argument('--report',type=Path,required=True)
    args=parser.parse_args()
    passwords=json.loads(args.credentials.read_text())['passwords']
    report={'passed':False,'scope':'Actual CPU Whisper-base, Tesseract OCR and TF-IDF/logistic-regression classifier on local OKD; fictional input only','checks':[]}
    def check(name,value):
        report['checks'].append({'name':name,'passed':bool(value)})
        print(('PASS ' if value else 'FAIL ')+name,flush=True)
        if not value:raise RuntimeError('Authenticated workflow acceptance failed: '+name)
    def login(api,actor):
        r=api.post('/api/auth/login',json={'username':actor,'password':passwords[actor]});r.raise_for_status()
        api.headers['X-Lab-CSRF']=r.json()['csrf']
    def completed(api,response):
        response.raise_for_status();job_id=response.json()['job_id']
        for _ in range(180):
            r=api.get('/api/jobs/'+job_id);r.raise_for_status();job=r.json()
            if job['status'] in {'completed','failed'}:break
            time.sleep(1)
        check('actual '+job['kind']+' job completed',job['status']=='completed')
        return job
    try:
        with PortForward('multimodal-ai-lab') as forward,httpx.Client(base_url=forward.base,timeout=300,trust_env=False) as api:
            login(api,'officer-a')
            status=api.get('/api/status').json()
            check('authenticated main lab retains speech OCR RAG monitoring',not status['simulated_identity'] and all(status['capabilities'][k] for k in ['speech_recognition','ocr','generative_llm','model_monitoring']))
            r=api.post('/api/cases',json={'title':'Fictional authenticated speech and scan exercise'});r.raise_for_status();case_id=r.json()['id'];report['case_id']=case_id
            completed(api,api.post('/api/cases/'+case_id+'/upload',data={'kind':'audio','language':'en'},
                files={'file':('fictional-statement.wav',args.fictional_audio.read_bytes(),'audio/wav')}))
            case=api.get('/api/cases/'+case_id).json()
            check('real speech produced timestamped original transcript',bool(case['segments']) and case['transcript_history'][-1]['engine']!='synthetic-fixture')
            original={s['id']:s['original_text'] for s in case['segments']}
            for index,segment in enumerate(case['segments']):
                text=segment['text']+(' The fictional bicycle was left beside the library.' if index==0 else '')
                r=api.patch('/api/cases/'+case_id+'/segments/'+segment['id'],json={'text':text,'expected_revision':case['revision']});r.raise_for_status();case=r.json()
            check('review preserves immutable original speech text',all(s['reviewed'] and s['original_text']==original[s['id']] for s in case['segments']))
            image=Image.new('RGB',(1400,240),'white');draw=ImageDraw.Draw(image)
            font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',46)
            draw.text((30,60),'Synthetic evidence inventory: blue bicycle.',font=font,fill='black')
            output=io.BytesIO();image.save(output,'PDF',resolution=150);image.close()
            completed(api,api.post('/api/cases/'+case_id+'/upload',data={'kind':'document'},files={'file':('fictional-scan.pdf',output.getvalue(),'application/pdf')}))
            case=api.get('/api/cases/'+case_id).json();doc=case['documents'][-1]
            check('actual scanned PDF uses local Tesseract and readable content','tesseract' in doc['engine'] and 'blue bicycle' in doc['text'].lower())
            r=api.post('/api/cases/'+case_id+'/documents/'+doc['id']+'/review',json={'text':doc['text'],'expected_revision':case['revision']});r.raise_for_status();case=r.json()
            r=api.post('/api/cases/'+case_id+'/draft');r.raise_for_status();case=r.json()
            check('officer cannot approve own draft',api.post('/api/cases/'+case_id+'/approve',json={'text':None,'expected_revision':case['revision']}).status_code==403)
            login(api,'reviewer');r=api.post('/api/cases/'+case_id+'/approve',json={'text':None,'expected_revision':case['revision']});r.raise_for_status();case=r.json()
            check('separate authenticated reviewer accepts reviewed draft',case['approved_by']=='reviewer' and case['approved_revision']==case['draft']['revision'] and case['status']=='approved')
            login(api,'officer-a');segment=case['segments'][0]
            r=api.patch('/api/cases/'+case_id+'/segments/'+segment['id'],json={'text':segment['text']+' A fictional correction was recorded.','expected_revision':case['revision']});r.raise_for_status();case=r.json()
            check('source correction invalidates earlier draft and approval',case['draft'] is None and case['approved_revision'] is None)
            login(api,'officer-b');check('another credential-bound case owner denied',api.get('/api/cases/'+case_id).status_code==403)
            check('officer model promotion refused',api.post('/api/models/promote',json={}).status_code==403)
            login(api,'reviewer');initial=api.get('/api/models').json()['active_version']
            if not initial:raise RuntimeError('An accepted active classifier must exist before preserving it')
            report['initial_active_version']=initial
            trained=completed(api,api.post('/api/models/train'));candidate=trained['result'];report['candidate']=candidate
            check('real classifier holds out eight examples and records MLflow artifact',candidate['training_samples']==24 and candidate['test_samples']==8 and candidate['gate_passed'] and candidate['artifact_published'] and candidate['mlflow_status']=='recorded_on_server')
            promoted=False
            try:
                r=api.post('/api/models/promote',json={'version':candidate['version']});r.raise_for_status();promoted=True
                check('authorized promotion selects verified candidate',r.json()['active_version']==candidate['version'])
                r=api.post('/api/models/predict',json={'text':'Evidence inventory of collected recording exhibits.'});r.raise_for_status()
                check('published classifier performs actual local inference',r.json().get('label')=='evidence_inventory')
            finally:
                if promoted:
                    r=api.post('/api/models/rollback');r.raise_for_status()
                    check('rollback restores original accepted active release',r.json()['active_version']==initial)
            report['passed']=True
    finally:
        args.report.parent.mkdir(parents=True,exist_ok=True)
        args.report.write_text(json.dumps(report,indent=2),encoding='utf-8');protect_file(args.report)


if __name__=='__main__':main()
