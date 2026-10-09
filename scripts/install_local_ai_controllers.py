"""Minimal free KServe Standard controllers on local CRC; no ODH full-stack claim."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
import yaml

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.local_cpe import Controller

FILES={'cert-manager-v1.21.2.yaml':'e03b668ec8675214af6b0a671699d088f2601fa3878e0dbe1b41d3feafd1879f',
       'kserve-v0.20.0.yaml':'6e6c1a5ceeb889e6ac41d68a551fa8c7d1fc1abcba10284154a0872bcc0d3fb8'}


def install(c,source):
    for name,expected in FILES.items():
        path=source/name
        if hashlib.sha256(path.read_bytes()).hexdigest()!=expected:raise RuntimeError('Official release manifest checksum changed')
    for ns in ['cert-manager','kserve','lab-ai-platform']:
        existing=c.oc('get','namespace',ns,'--ignore-not-found','-o','json').stdout.strip()
        if existing and json.loads(existing)['metadata'].get('labels',{}).get('lab.multimodal.local/ai-platform')!='true':
            raise RuntimeError('Refusing to change an existing namespace not owned by this local exercise: '+ns)
        c.apply({'apiVersion':'v1','kind':'Namespace','metadata':{'name':ns,'labels':{'lab.multimodal.local/ai-platform':'true'}}})
    for name in FILES:
        docs=list(yaml.load_all((source/name).read_text(),Loader=getattr(yaml,'CSafeLoader',yaml.SafeLoader)))
        for obj in docs:
            if not obj:continue
            kind=obj['kind'];metadata=obj['metadata'];resource=metadata['name']
            if kind=='Namespace':continue
            if name.startswith('kserve') and any(x in resource for x in ['llmisvc','llminference','localmodel']):
                if kind!='CustomResourceDefinition':continue
            if kind=='DaemonSet':continue
            if kind=='Deployment':
                pod=obj['spec']['template']['spec'];pod['securityContext']={'runAsNonRoot':True,'seccompProfile':{'type':'RuntimeDefault'}}
                for container in pod['containers']:
                    container['securityContext']={'allowPrivilegeEscalation':False,'capabilities':{'drop':['ALL']},'readOnlyRootFilesystem':True}
                    container['resources']={'requests':{'cpu':'50m','memory':'128Mi'},'limits':{'cpu':'500m','memory':'512Mi'}}
                if resource=='kserve-controller-manager':
                    pod['containers']=[x for x in pod['containers'] if x['name']=='manager']
                    # Operator-owned predictive Standard deployments only in this lab project.
                    pod['containers'][0]['env']=pod['containers'][0].get('env',[])+[{'name':'KSERVE_ENABLE_SELF_SIGNED_CA','value':'false'}]
            if kind=='ConfigMap' and resource=='inferenceservice-config':
                deploy=json.loads(obj['data']['deploy']);deploy['defaultDeploymentMode']='Standard';obj['data']['deploy']=json.dumps(deploy)
                ingress=json.loads(obj['data']['ingress']);ingress.update(disableIngressCreation=True,enableGatewayApi=False,disableIstioVirtualHost=True);obj['data']['ingress']=json.dumps(ingress)
            if kind=='CustomResourceDefinition':
                c.oc('apply','--server-side','--field-manager=local-ai-platform','-f','-',payload=obj)
            else:c.apply(obj)
        if name.startswith('cert'):
            for deploy in ['cert-manager','cert-manager-cainjector','cert-manager-webhook']:
                c.oc('rollout','status','deployment/'+deploy,'-n','cert-manager','--timeout=240s')
    c.oc('rollout','status','deployment/kserve-controller-manager','-n','kserve','--timeout=240s')
    c.save('controllers-install.json',{'scope':'KServe Standard predictive controller; no Knative/Istio/ODH dashboard/GPU',
                                      'manifests_sha256':FILES,'installed':True})
    print('Local cert-manager and KServe Standard controllers ready. Model serving acceptance remains required.')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--admin-kubeconfig',type=Path,required=True);p.add_argument('--private-dir',type=Path,required=True)
    args=p.parse_args();c=Controller(args.admin_kubeconfig,args.private_dir);install(c,args.private_dir)

if __name__=='__main__':main()
