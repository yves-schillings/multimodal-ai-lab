"""Actual two-project RAG isolation, lifecycle and database restart acceptance."""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.local_cpe import Controller,PROJECTS
from scripts.cluster_cpe_acceptance import verify,forward,login,phase
from scripts.cluster_cpe_lifecycle_acceptance import accept


def accept_all(c):
    for key in ['a','b']:
        for prefix in ['verified','lifecycle']:
            old=c.private/(prefix+'-'+key+'.json')
            archived=c.private/(prefix+'-'+key+'-before-rag.json')
            if old.exists() and not archived.exists():archived.write_bytes(old.read_bytes())
        report=verify(c,key);c.save('verified-'+key+'.json',report)
        lifecycle=accept(c,key);c.save('lifecycle-'+key+'.json',lifecycle)
        with forward(c,key) as api:
            api.timeout=300
            phase(api,'active')
            passwords=json.loads((c.private/('credentials-'+key+'.json')).read_text())['passwords']
            login(api,'reviewer',passwords)
            response=api.post('/api/cases/'+report['case_id']+'/question',json={'question':'Where was the bicycle left?'})
            response.raise_for_status()
            if response.json()['abstained'] or 'museum' not in response.json()['answer'].lower():
                raise RuntimeError('CPE RAG failed after application replacement')
        print('PASS CPE '+key+' RAG after application replacement',flush=True)
    before=c.get('pvc','lab-vector-data','multimodal-ai-lab')['metadata']['uid']
    c.oc('delete','pod','lab-vector-0','-n','multimodal-ai-lab')
    c.oc('rollout','status','statefulset/lab-vector','-n','multimodal-ai-lab','--timeout=240s')
    checks=[]
    for key in ['a','b']:
        report=json.loads((c.private/('verified-'+key+'.json')).read_text())
        with forward(c,key) as api:
            api.timeout=300;phase(api,'active')
            login(api,'reviewer',json.loads((c.private/('credentials-'+key+'.json')).read_text())['passwords'])
            response=api.post('/api/cases/'+report['case_id']+'/question',json={'question':'Where was the bicycle left?'})
            response.raise_for_status();answer=response.json()
            passed=not answer['abstained'] and 'museum' in answer['answer'].lower()
            checks.append({'name':'CPE '+key+' RAG persists after vector database replacement','passed':passed})
            if not passed:raise RuntimeError(checks[-1]['name'])
        print('PASS '+checks[-1]['name'],flush=True)
    checks.append({'name':'Original shared vector PVC UID retained','passed':before==c.get('pvc','lab-vector-data','multimodal-ai-lab')['metadata']['uid']})
    passed=all(x['passed'] for x in checks)
    c.save('rag-restart.json',{'passed':passed,'checks':checks,'verified_at':time.time()})
    if not passed:raise RuntimeError('Vector storage identity changed during acceptance')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--admin-kubeconfig',type=Path,required=True);p.add_argument('--private-dir',type=Path,required=True)
    args=p.parse_args();c=Controller(args.admin_kubeconfig,args.private_dir)
    try:
        accept_all(c)
    except Exception:
        # A later persistence failure must not leave a partially accepted AI profile active.
        revoked=[]
        for key in ['a','b']:
            try:c.gate(key,'revoked');revoked.append(key)
            except Exception:pass  # Retain the original failure; record any unreachable gate.
        c.save('rag-acceptance-failure.json',{'passed':False,'revoked_projects':revoked,'failed_at':time.time()})
        raise

if __name__=='__main__':main()
