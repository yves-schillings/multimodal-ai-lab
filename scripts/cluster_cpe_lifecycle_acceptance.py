"""Accept fresh evidence, exercise revocation/expiry, and retain isolated CPE data."""
import argparse
import copy
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.cluster_cpe_acceptance import forward,login,phase
from scripts.local_cpe import Controller,PROJECTS,digest


def accept(controller,key):
    ns=controller.namespace(key);checks=[];passed=False
    def check(name,value):
        checks.append({'name':name,'passed':bool(value)})
        print(('PASS ' if value else 'FAIL ')+name,flush=True)
        if not value:raise RuntimeError('CPE lifecycle acceptance failed: '+name)
    path=controller.private/('verified-'+key+'.json')
    verified=json.loads(path.read_text());original=path.read_bytes()
    passwords=json.loads((controller.private/('credentials-'+key+'.json')).read_text())['passwords']
    before=controller.snapshot(key)
    try:
        # Deliberately invalid evidence cannot activate, even with an operator command.
        invalid=copy.deepcopy(verified);invalid['resource_digest']='0'*64
        controller.save(path.name,invalid)
        refused=False
        try:controller.activate(key)
        except RuntimeError:refused=True
        finally:path.write_bytes(original)
        check('activation rejects evidence for different resources',refused)
        controller.activate(key)
        with forward(controller,key) as api:
            phase(api,'active');login(api,PROJECTS[key],passwords)
            check('verified active project admits its own account',api.get('/api/cases').status_code==200)
            created=api.post('/api/cases',json={'title':'Synthetic CPE persistent lifecycle'});created.raise_for_status()
            case_id=created.json()['id']
            check('active case belongs to authenticated project actor',created.json()['owner_actor']==PROJECTS[key])
            check('active project still rejects unapproved speech processing',api.post('/api/cases/'+case_id+'/upload',
                files={'file':('fictional.wav',b'no-approved-speech-model','audio/wav')},data={'kind':'audio'}).status_code==403)
            check('active project still rejects model release actions',api.get('/api/models').status_code==403)
            check('other actor cannot select this session',api.get('/api/cases/'+case_id,params={'actor':'reviewer'}).status_code==403)
            controller.gate(key,'revoked');phase(api,'revoked')
            check('revocation denies existing authenticated case access',api.get('/api/cases/'+case_id).status_code==503)
            current=controller.get('configmap','cpe-control',ns)
            control={'apiVersion':'v1','kind':'ConfigMap','metadata':{'name':'cpe-control','namespace':ns},'data':{}}
            data=json.loads(current['data']['control.json']);data['phase']='active';data['expires']=time.time()-1
            data['evidence_sha256']=hashlib.sha256(original).hexdigest()
            control['data']['control.json']=json.dumps(data);controller.apply(control);controller.refresh_control(key)
            phase(api,'expired')
            check('expiry denies existing authenticated case access',api.get('/api/cases/'+case_id).status_code==503)
            control['data']['control.json']='[]';controller.apply(control);controller.refresh_control(key)
            phase(api,'unavailable')
            check('unreadable activation control fails closed',api.get('/api/cases/'+case_id).status_code==503)
            controller.activate(key);phase(api,'active')
            token=api.cookies.get('lab_session');csrf=api.headers['X-Lab-CSRF']
        controller.oc('delete','pod','-n',ns,'-l','app=cpe-app')
        controller.oc('rollout','status','deployment/cpe-app','-n',ns,'--timeout=180s')
        with forward(controller,key) as api:
            api.cookies.set('lab_session',token);api.headers['X-Lab-CSRF']=csrf
            check('authenticated session persists after project pod replacement',api.get('/api/auth/me').status_code==200)
            response=api.get('/api/cases/'+case_id)
            check('isolated case persists after project pod replacement',response.status_code==200 and response.json()['id']==case_id)
            check('activation survives project pod replacement',api.get('/api/status').json()['cpe']['active'])
        after=controller.snapshot(key)
        check('all protected project resources and storage UID retained',digest(before)==digest(after))
        passed=True
        return {'passed':True,'checks':checks,'namespace':ns,'case_id':case_id,'accepted_at':time.time(),
            'image':after['deployment/cpe-app']['spec']['template']['spec']['containers'][0]['image'],
            'scope':'Actual local OKD CPE lifecycle; model scope follows the verified profile; shared hostpath storage admission budget only'}
    finally:
        if not passed:controller.gate(key,'revoked')
        controller.save('lifecycle-progress-'+key+'.json',{'passed':passed,'checks':checks})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project',choices=PROJECTS,required=True)
    parser.add_argument('--admin-kubeconfig',type=Path,required=True)
    parser.add_argument('--private-dir',type=Path,required=True)
    args=parser.parse_args();controller=Controller(args.admin_kubeconfig,args.private_dir)
    report=accept(controller,args.project);controller.save('lifecycle-'+args.project+'.json',report)
    print('Actual local CPE activation, denial and persistence acceptance passed.')


if __name__=='__main__':main()
