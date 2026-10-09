"""Deploy lightweight Jupyter workbench, real CPU training Job and KServe classifier."""
import argparse
import base64
import json
import secrets
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.local_cpe import Controller


def deploy(c):
    ns='lab-ai-platform';main='multimodal-ai-lab'
    image=c.get('istag','lab-app:latest',main)['image']['dockerImageReference']
    c.apply({'apiVersion':'v1','kind':'ServiceAccount','metadata':{'name':'lab-ai-runtime','namespace':ns},'automountServiceAccountToken':False})
    # Keep the existing MLflow allowlisted Host header; DNS alias stays inside CRC.
    c.apply({'apiVersion':'v1','kind':'Service','metadata':{'name':'lab-mlflow','namespace':ns},
             'spec':{'type':'ExternalName','externalName':'lab-mlflow.multimodal-ai-lab.svc.cluster.local','ports':[{'port':5000}]}})
    c.apply({'apiVersion':'rbac.authorization.k8s.io/v1','kind':'RoleBinding','metadata':{'name':'lab-ai-platform-image-pull','namespace':main},
       'roleRef':{'apiGroup':'rbac.authorization.k8s.io','kind':'ClusterRole','name':'system:image-puller'},
       'subjects':[{'kind':'ServiceAccount','name':'lab-ai-runtime','namespace':ns}]})
    c.apply({'apiVersion':'v1','kind':'ResourceQuota','metadata':{'name':'platform-budget','namespace':ns},'spec':{'hard':{
      'requests.cpu':'800m','limits.cpu':'3','requests.memory':'2Gi','limits.memory':'4Gi',
      'requests.storage':'2Gi','persistentvolumeclaims':'2','pods':'4','services.nodeports':'0','services.loadbalancers':'0','requests.nvidia.com/gpu':'0'}}})
    existing=c.oc('get','secret','platform-access','-n',ns,'--ignore-not-found','-o','json').stdout.strip()
    tokens={k:base64.b64decode(v).decode() for k,v in json.loads(existing)['data'].items()} if existing else {'token':secrets.token_urlsafe(36),'jupyter':secrets.token_urlsafe(36)}
    c.apply({'apiVersion':'v1','kind':'Secret','metadata':{'name':'platform-access','namespace':ns},'stringData':tokens})
    c.save('workbench-access.json',tokens)
    for name in ['workbench-data','training-results']:
        c.apply({'apiVersion':'v1','kind':'PersistentVolumeClaim','metadata':{'name':name,'namespace':ns},'spec':{'accessModes':['ReadWriteOnce'],'resources':{'requests':{'storage':'1Gi'}}}})
    c.apply({'apiVersion':'networking.k8s.io/v1','kind':'NetworkPolicy','metadata':{'name':'platform-default-deny','namespace':ns},
        'spec':{'podSelector':{},'policyTypes':['Ingress','Egress'],'ingress':[],'egress':[]}})
    c.apply({'apiVersion':'networking.k8s.io/v1','kind':'NetworkPolicy','metadata':{'name':'platform-local','namespace':ns},
        'spec':{'podSelector':{},'policyTypes':['Ingress','Egress'],
         'ingress':[{'from':[{'podSelector':{'matchLabels':{'app':'lab-workbench'}}}],'ports':[{'port':8080}]}],
         'egress':[{'to':[{'podSelector':{}}],'ports':[{'port':8080}]},
           {'to':[{'namespaceSelector':{'matchLabels':{'kubernetes.io/metadata.name':main}},'podSelector':{'matchLabels':{'app.kubernetes.io/name':'lab-mlflow'}}}], 'ports':[{'port':5000}]},
           {'to':[{'namespaceSelector':{'matchLabels':{'kubernetes.io/metadata.name':'openshift-dns'}},'podSelector':{'matchLabels':{'dns.operator.openshift.io/daemonset-dns':'default'}}}], 'ports':[{'port':5353,'protocol':'UDP'},{'port':5353,'protocol':'TCP'}]}]}})
    c.apply({'apiVersion':'networking.k8s.io/v1','kind':'NetworkPolicy','metadata':{'name':'platform-training-mlflow','namespace':main},
        'spec':{'podSelector':{'matchLabels':{'app.kubernetes.io/name':'lab-mlflow'}},'policyTypes':['Ingress'],
        'ingress':[{'from':[{'namespaceSelector':{'matchLabels':{'kubernetes.io/metadata.name':ns}},'podSelector':{'matchLabels':{'app':'lab-training'}}}], 'ports':[{'port':5000}]}]}})
    security={'allowPrivilegeEscalation':False,'readOnlyRootFilesystem':True,'capabilities':{'drop':['ALL']}}
    podbase={'serviceAccountName':'lab-ai-runtime','automountServiceAccountToken':False,'enableServiceLinks':False,
      'securityContext':{'runAsNonRoot':True,'seccompProfile':{'type':'RuntimeDefault'}}}
    # The accepted official Python 3.12 image stays fixed across later reruns.
    notebook='quay.io/jupyter/minimal-notebook@sha256:d65a4aef5c169c199e91232b47932c283fda2dc56ba157967def4d05f7b4e408'
    workbench={**podbase,'containers':[{'name':'jupyter','image':notebook,
      'command':['python','-m','jupyterlab','--ip=0.0.0.0','--port=8888','--no-browser','--ServerApp.root_dir=/work','--ServerApp.allow_remote_access=True'],
      'env':[{'name':k,'value':v} for k,v in {'HOME':'/work','JUPYTER_CONFIG_DIR':'/tmp/config','JUPYTER_RUNTIME_DIR':'/tmp/runtime','JUPYTER_DATA_DIR':'/tmp/jupyter','JUPYTER_TOKEN_FILE':'/credentials/jupyter'}.items()],
      'resources':{'requests':{'cpu':'100m','memory':'512Mi'},'limits':{'cpu':'1','memory':'1Gi'}},'securityContext':security,
      'readinessProbe':{'httpGet':{'path':'/api','port':8888}},
      'volumeMounts':[{'name':'work','mountPath':'/work'},{'name':'tmp','mountPath':'/tmp'},{'name':'credentials','mountPath':'/credentials','readOnly':True},{'name':'results','mountPath':'/results','readOnly':True}]}],
      'volumes':[{'name':'work','persistentVolumeClaim':{'claimName':'workbench-data'}},{'name':'tmp','emptyDir':{'sizeLimit':'128Mi'}},{'name':'credentials','secret':{'secretName':'platform-access','defaultMode':288}},{'name':'results','persistentVolumeClaim':{'claimName':'training-results'}}]}
    c.apply({'apiVersion':'apps/v1','kind':'Deployment','metadata':{'name':'lab-workbench','namespace':ns},'spec':{'replicas':0,'strategy':{'type':'Recreate'},'selector':{'matchLabels':{'app':'lab-workbench'}},'template':{'metadata':{'labels':{'app':'lab-workbench'}},'spec':workbench}}})
    c.apply({'apiVersion':'v1','kind':'Service','metadata':{'name':'lab-workbench','namespace':ns},'spec':{'selector':{'app':'lab-workbench'},'ports':[{'port':8888}]}})
    c.apply({'apiVersion':'v1','kind':'ConfigMap','metadata':{'name':'platform-code','namespace':ns},'data':{
       'predictor.py':(ROOT/'scripts/platform_predictor.py').read_text(),'train.py':(ROOT/'scripts/platform_training.py').read_text()}})
    # Read accepted active metadata/artifact inside the existing app, without changing its release.
    pods=json.loads(c.oc('get','pods','-n',main,'-l','app.kubernetes.io/name=lab-app','-o','json').stdout)['items']
    apppod=next(x['metadata']['name'] for x in pods if x['status']['phase']=='Running')
    source="import json,base64;from pathlib import Path;from lab.store import StatementStore;from lab.models import ModelRegistry;r=ModelRegistry(StatementStore(Path('/app/data')));s=r.state();m=next(v for v in s['versions'] if v['version']==s['active_version']);r._verify_release(m);print(json.dumps({'metadata':m,'artifact':base64.b64encode((r.root/(m['version']+'.pkl')).read_bytes()).decode()}))"
    snapshot=json.loads(c.oc('exec','-n',main,apppod,'--','python','-c',source).stdout)
    c.apply({'apiVersion':'v1','kind':'ConfigMap','metadata':{'name':'accepted-classifier','namespace':ns},
       'data':{'metadata.json':json.dumps(snapshot['metadata'])},'binaryData':{'classifier.pkl':snapshot['artifact']}})
    predictor={**podbase,'containers':[{'name':'kserve-container','image':image,'command':['python','-m','uvicorn','predictor:app','--app-dir','/code','--host','0.0.0.0','--port','8080','--no-access-log'],
       'resources':{'requests':{'cpu':'100m','memory':'256Mi'},'limits':{'cpu':'1','memory':'1Gi'}},'securityContext':security,'ports':[{'containerPort':8080}],
       'readinessProbe':{'httpGet':{'path':'/v2/health/ready','port':8080}},
       'volumeMounts':[{'name':'code','mountPath':'/code','readOnly':True},{'name':'model','mountPath':'/model','readOnly':True},{'name':'credentials','mountPath':'/credentials','readOnly':True}]}],
       'volumes':[{'name':'code','configMap':{'name':'platform-code'}},{'name':'model','configMap':{'name':'accepted-classifier'}},{'name':'credentials','secret':{'secretName':'platform-access','defaultMode':288}}]}
    c.apply({'apiVersion':'serving.kserve.io/v1beta1','kind':'InferenceService','metadata':{'name':'document-classifier','namespace':ns,
      'annotations':{'serving.kserve.io/deploymentMode':'Standard','serving.kserve.io/autoscalerClass':'external'}},'spec':{'predictor':{**predictor,'minReplicas':1,'maxReplicas':1}}})
    jobname='classifier-training-'+str(int(time.time()))
    training={**podbase,'restartPolicy':'Never','containers':[{'name':'train','image':image,'command':['python','/code/train.py'],
       'env':[{'name':'MLFLOW_TRACKING_URI','value':'http://lab-mlflow:5000'}],
       'resources':{'requests':{'cpu':'500m','memory':'512Mi'},'limits':{'cpu':'1','memory':'1Gi'}},'securityContext':security,
       'volumeMounts':[{'name':'code','mountPath':'/code','readOnly':True},{'name':'results','mountPath':'/results'},{'name':'tmp','mountPath':'/tmp'}]}],
       'volumes':[{'name':'code','configMap':{'name':'platform-code'}},{'name':'results','persistentVolumeClaim':{'claimName':'training-results'}},{'name':'tmp','emptyDir':{'sizeLimit':'256Mi'}}]}
    c.apply({'apiVersion':'batch/v1','kind':'Job','metadata':{'name':jobname,'namespace':ns},'spec':{'backoffLimit':0,'activeDeadlineSeconds':240,'template':{'metadata':{'labels':{'app':'lab-training'}},'spec':training}}})
    c.save('workloads.json',{'image':image,'jupyter_image':notebook,'training_job':jobname,'accepted_classifier':snapshot['metadata']['version'],'model_sha256':snapshot['metadata']['artifact_sha256']})
    c.oc('wait','--for=condition=complete','job/'+jobname,'-n',ns,'--timeout=240s')
    c.oc('scale','deployment/lab-workbench','--replicas=1','-n',ns)
    print('Local workbench, training Job and KServe InferenceService submitted. Actual acceptance pending.',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--admin-kubeconfig',type=Path,required=True);p.add_argument('--private-dir',type=Path,required=True)
    args=p.parse_args();deploy(Controller(args.admin_kubeconfig,args.private_dir))

if __name__=='__main__':main()
