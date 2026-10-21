"""Operator-managed local synthetic CPEs. Never changes the existing lab volumes."""
import argparse
import base64
import hashlib
import json
import os
import secrets
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import urlsplit

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from lab.auth import password_hash
from scripts.deploy_local_auth import protect_file

LABEL={'lab.multimodal.local/cpe':'true'}
PROJECTS={'a':'officer-a','b':'officer-b'}


def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


class Controller:
    def __init__(self,kubeconfig,private):
        self.kubeconfig=str(kubeconfig.resolve());self.private=private.resolve()
        if self.private==ROOT or ROOT in self.private.parents:raise ValueError('Private records must be outside Git')
        self.private.mkdir(parents=True,exist_ok=True)
        server=self.oc('whoami','--show-server').stdout.strip()
        if urlsplit(server).hostname!='api.crc.testing':raise RuntimeError('Only the local CRC cluster is permitted')
        if self.oc('auth','can-i','create','namespaces').stdout.strip()!='yes':raise RuntimeError('Local operator permissions required')

    def oc(self,*args,check=True,payload=None,timeout=300):
        r=subprocess.run(['oc','--kubeconfig',self.kubeconfig,*args],input=json.dumps(payload) if payload is not None else None,
                         capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=timeout)
        if check and r.returncode:raise RuntimeError('Local cluster operation failed: '+' '.join(args[:3])+': '+r.stderr[-1200:])
        return r

    def get(self,kind,name,ns=None):
        args=['get',kind,name,'-o','json']+(['-n',ns] if ns else [])
        return json.loads(self.oc(*args).stdout)

    # Each Python dictionary below describes a Kubernetes object, not a model.
    # apiVersion chooses the Kubernetes API group and schema. kind names the object.
    # metadata identifies it. spec describes the configuration requested from OKD.
    # oc apply -f - sends this dictionary as JSON through standard input. The API
    # server validates and stores it; the relevant cluster component enforces it.
    # The operator runs this method. The application receives no operator token.
    def apply(self,obj):self.oc('apply','-f','-',payload=obj)

    def save(self,name,data):
        path=self.private/name
        path.write_text(json.dumps(data,indent=2),encoding='utf-8');protect_file(path)
        return path

    def namespace(self,key):
        if key not in PROJECTS:raise ValueError('Choose only synthetic project a or b')
        return 'lab-cpe-'+key

    def gate(self,key,phase,evidence=None):
        ns=self.namespace(key);profile=json.loads(self.get('configmap','cpe-profile',ns)['data']['profile.json'])
        gate={'schema':1,'phase':phase,'approval':profile['approval'],'expires':profile['expires'],
              'models':profile['models'],'profile_digest':digest(profile),'evidence_sha256':evidence}
        self.apply({'apiVersion':'v1','kind':'ConfigMap','metadata':{'name':'cpe-control','namespace':ns},
                    'data':{'control.json':json.dumps(gate)}})
        self.refresh_control(key)

    def refresh_control(self,key):

        # Pod annotation updates prompt kubelet to refresh mounted ConfigMaps.
        # The caller still waits for the application's observed phase before acceptance.
        self.oc('annotate','pods','-n',self.namespace(key),'-l','app=cpe-app',
                'lab.multimodal.local/control-refresh='+str(time.time_ns()),'--overwrite')

    def provision(self,key,approval):
        ns=self.namespace(key)
        existing=self.oc('get','namespace',ns,'--ignore-not-found','-o','name').stdout.strip()
        if existing:
            known=self.get('namespace',ns)
            if known['metadata'].get('labels',{}).get('lab.multimodal.local/cpe')!='true':
                raise RuntimeError('Existing namespace is not owned by this CPE workflow')
            if self.oc('get','deployment','cpe-app','-n',ns,'--ignore-not-found','-o','name').stdout.strip():
                raise RuntimeError('Existing runtime retained. Use verify/activate/revoke, not reprovision')
        if not approval.strip():raise ValueError('An explicit synthetic owner approval reference is required')
        image=self.get('istag','lab-app:latest','multimodal-ai-lab')['image']['dockerImageReference']
        if '@sha256:' not in image:raise ValueError('An immutable local app image is required')
        profile={'key':key,'namespace':ns,'image':image,'approval':approval,'expires':time.time()+86400,
                 'models':[],'scope':'synthetic_documents_lexical_only','storage':'1Gi admission request; shared hostpath physical capacity',
                 'runtime_cpu':'1','runtime_memory':'1Gi','project_actor':PROJECTS[key]}
        if existing:
            previous=self.oc('get','configmap','cpe-profile','-n',ns,'--ignore-not-found','-o','json').stdout.strip()
            if previous:
                profile=json.loads(json.loads(previous)['data']['profile.json'])
                if profile['approval']!=approval:raise RuntimeError('Pending owner approval must be preserved')
                image=profile['image']

        # PROJECT BOUNDARY: ns is lab-cpe-a or lab-cpe-b. A namespace groups this
        # project's objects. It does not, by itself, block network traffic or
        # decide which human may read a case. The following controls add those
        # separate protections. Restricted pod-security labels request pod checks.
        self.apply({'apiVersion':'v1','kind':'Namespace','metadata':{'name':ns,'labels':{**LABEL,
                    'pod-security.kubernetes.io/enforce':'restricted','pod-security.kubernetes.io/audit':'restricted'}}})
        self.admission(ns,image)

        # CONTROL 3 - PROJECT BUDGET: admission rejects new objects whose declared
        # resource requests/limits exceed the namespace totals. 500m means half
        # of one CPU core; Mi/Gi are binary memory/storage units. GPU requests,
        # external LoadBalancer services and NodePort services have a zero budget.
        # A 1Gi storage REQUEST is not a physical filesystem quota on CRC hostpath.
        self.apply({'apiVersion':'v1','kind':'ResourceQuota','metadata':{'name':'cpe-budget','namespace':ns},'spec':{'hard':{
            'requests.cpu':'500m','limits.cpu':'1','requests.memory':'512Mi','limits.memory':'1Gi',
            'requests.storage':'1Gi','persistentvolumeclaims':'1','pods':'2','secrets':'8','configmaps':'4',
            'requests.nvidia.com/gpu':'0','services.loadbalancers':'0','services.nodeports':'0'}}})

        # CONTROL 3 - CONTAINER DEFAULTS: LimitRange supplies defaults when a
        # container omits requests/limits, and rejects values above its maximum.
        # Requests guide scheduling. CPU limits can throttle execution; exceeding
        # a memory limit can terminate a container. ResourceQuota still applies.
        self.apply({'apiVersion':'v1','kind':'LimitRange','metadata':{'name':'cpe-container-limits','namespace':ns},'spec':{'limits':[{
            'type':'Container','max':{'cpu':'1','memory':'1Gi'},'default':{'cpu':'1','memory':'1Gi'},
            'defaultRequest':{'cpu':'500m','memory':'512Mi'}}]}})

        # CONTROL 1 - WORKLOAD IDENTITY: create a technical account for the pod.
        # This account identifies application processes to Kubernetes. It is NOT
        # the Officer's login account. Human logins and case rights are checked
        # separately by lab/auth.py and the application's case store.
        self.apply({

            # v1 is the stable CORE Kubernetes API schema for ServiceAccount.
            # It is not the Python version or the AI model version.
            'apiVersion': 'v1',

            # kind tells the API server which type of object to create/update.
            'kind': 'ServiceAccount',

            # The same account name may exist in two different namespaces.
            # Its full identity includes the namespace, so these are distinct.
            'metadata': {
                'name': 'cpe-runtime',
                'namespace': ns,
            },

            # Do not automatically put a Kubernetes API access token in the pod.
            # The application does not need to administer cluster resources.
            # The Deployment below repeats this restriction on its pod template.
            'automountServiceAccountToken': False,
        })

        # CONTROL 1 - IMAGE DOWNLOAD PERMISSION: the project's pod needs the
        # approved container image held by the main lab's internal registry.
        # This binding grants that narrow registry permission. It does not grant
        # access to another project's cases, documents, Secrets or vector database.
        self.apply({

            # RBAC means Role-Based Access Control. This object belongs to the
            # RBAC API group, whereas ServiceAccount above uses the core API.
            'apiVersion': 'rbac.authorization.k8s.io/v1',
            'kind': 'RoleBinding',

            # A RoleBinding applies in the namespace WHERE THE RESOURCE LIVES.
            # Here the images are in multimodal-ai-lab, not in lab-cpe-a/b.
            # The generated binding name keeps project A and B grants separate.
            'metadata': {
                'name': ns + '-image-pull',
                'namespace': 'multimodal-ai-lab',
            },

            # roleRef selects the existing OpenShift image-puller role. Using a
            # ClusterRole here reuses its permission definition. A RoleBinding
            # still limits this grant to its own namespace; it is not a
            # ClusterRoleBinding and does not make the workload a cluster admin.
            'roleRef': {
                'apiGroup': 'rbac.authorization.k8s.io',
                'kind': 'ClusterRole',
                'name': 'system:image-puller',
            },

            # subjects identifies WHO receives this permission: only the technical
            # cpe-runtime account in the selected project's namespace. No human
            # account and no service account from another namespace is included.
            'subjects': [{
                'kind': 'ServiceAccount',
                'name': 'cpe-runtime',
                'namespace': ns,
            }],
        })

        # CONTROL 2 - DEFAULT DENY: an empty podSelector selects every pod in ns.
        # Empty ingress and egress lists allow no traffic under THIS policy.
        # NetworkPolicy allow rules are additive, so inspect all policies too.
        # The cluster network plugin enforces traffic rules, not this Python code.
        # Operator port-forward is a separate path and bypasses NetworkPolicy.
        self.apply({'apiVersion':'networking.k8s.io/v1','kind':'NetworkPolicy','metadata':{'name':'cpe-default-deny','namespace':ns},
                    'spec':{'podSelector':{},'policyTypes':['Ingress','Egress'],'ingress':[],'egress':[]}})

        # A controlled in-namespace verification client may reach the app. There is no external Route or model egress.
        self.apply({'apiVersion':'networking.k8s.io/v1','kind':'NetworkPolicy','metadata':{'name':'cpe-local-http','namespace':ns},
                    'spec':{'podSelector':{'matchLabels':{'app':'cpe-app'}},'policyTypes':['Ingress','Egress'],
                            'ingress':[{'from':[{'podSelector':{'matchLabels':{'app':'cpe-app'}}}],'ports':[{'port':8770}]}],
                            'egress':[{'to':[{'podSelector':{'matchLabels':{'app':'cpe-app'}}}],'ports':[{'port':8770}]}]}})

        # APPROVAL PROFILE: ConfigMap stores non-secret scope, expiry, image and
        # allowed models. The separate control ConfigMap starts as provisioned,
        # not active. The application gate in lab/cpe.py refuses normal use until
        # verification succeeds and the operator explicitly activates the project.
        self.apply({'apiVersion':'v1','kind':'ConfigMap','metadata':{'name':'cpe-profile','namespace':ns},'data':{'profile.json':json.dumps(profile)}})
        self.gate(key,'provisioned')

        # CONTROL 4 - HUMAN LOGIN CREDENTIALS: keep generated clear passwords in
        # protected operator records outside Git. Only password hashes enter the
        # application's Secret. Hashing is performed by lab.auth.password_hash.
        credential_path=self.private/('credentials-'+key+'.json')
        if credential_path.exists():
            passwords=json.loads(credential_path.read_text(encoding='utf-8'))['passwords']
        else:passwords={actor:secrets.token_urlsafe(24) for actor in (PROJECTS[key],'reviewer')}
        accounts={actor:{'enabled':True,'password_hash':password_hash(password)} for actor,password in passwords.items()}
        self.save('credentials-'+key+'.json',{'passwords':passwords})
        if not self.oc('get','secret','cpe-credentials','-n',ns,'--ignore-not-found','-o','name').stdout.strip():

            # Opaque means application-defined Secret data. Kubernetes requires
            # base64 for this data field; base64 is encoding, NOT encryption.
            # API permissions and cluster storage protection remain necessary.
            self.apply({'apiVersion':'v1','kind':'Secret','metadata':{'name':'cpe-credentials','namespace':ns},'type':'Opaque',
                        'data':{'credentials.json':base64.b64encode(json.dumps(accounts).encode()).decode()}})

        # CONTROL 5 - PERSISTENCE: a PersistentVolumeClaim (PVC) requests a volume
        # for this project. Pod replacement reuses it instead of losing cases,
        # originals, reviewed text and audit records. ReadWriteOnce is a volume
        # access mode, not a human access rule. CRC volumes share physical disk.
        self.apply({'apiVersion':'v1','kind':'PersistentVolumeClaim','metadata':{'name':'cpe-data','namespace':ns},'spec':{
            'accessModes':['ReadWriteOnce'],'resources':{'requests':{'storage':'1Gi'}}}})

        # WORKLOAD CONTRACT: the pod uses the technical account defined above,
        # runs as a non-root user and drops Linux capabilities. Its root filesystem
        # is read-only. Only data and temporary mounts are writable. Secret and
        # approval files are mounted read-only; no Kubernetes API token is mounted.
        # CONTROL 6 BASELINE: lexical retrieval is enabled, but models=[] approves
        # no model endpoints. extend_cpe_rag.py adds the bounded offline AI profile.
        pod={'metadata':{'labels':{'app':'cpe-app'}},'spec':{'serviceAccountName':'cpe-runtime','automountServiceAccountToken':False,
             'enableServiceLinks':False,'securityContext':{'runAsNonRoot':True,'seccompProfile':{'type':'RuntimeDefault'}},
             'containers':[{'name':'app','image':image,'ports':[{'containerPort':8770}],
                'env':[{'name':k,'value':v} for k,v in {
                    'LAB_BIND_HOST':'0.0.0.0','LAB_NETWORK_MODE':'container','LAB_AUTH_MODE':'local_sessions',
                    'LAB_CREDENTIAL_FILE':'/app/secrets/identity/credentials.json','LAB_CPE_STATE_FILE':'/app/control/control.json',
                    'LAB_MONITORING':'disabled','LAB_RETRIEVAL_BACKEND':'lexical'}.items()],
                'resources':{'requests':{'cpu':'500m','memory':'512Mi'},'limits':{'cpu':'1','memory':'1Gi'}},
                'securityContext':{'allowPrivilegeEscalation':False,'readOnlyRootFilesystem':True,'capabilities':{'drop':['ALL']}},
                'volumeMounts':[{'name':'data','mountPath':'/app/data'},{'name':'tmp','mountPath':'/tmp'},
                    {'name':'identity','mountPath':'/app/secrets/identity','readOnly':True},
                    {'name':'control','mountPath':'/app/control','readOnly':True}],
                'readinessProbe':{'httpGet':{'path':'/health/ready','port':8770,'httpHeaders':[{'name':'Host','value':'127.0.0.1'}]},'periodSeconds':5},
                'livenessProbe':{'httpGet':{'path':'/health/live','port':8770,'httpHeaders':[{'name':'Host','value':'127.0.0.1'}]},'periodSeconds':30}}],
             'volumes':[{'name':'data','persistentVolumeClaim':{'claimName':'cpe-data'}},{'name':'tmp','emptyDir':{'sizeLimit':'64Mi'}},
                {'name':'identity','secret':{'secretName':'cpe-credentials','defaultMode':288}},
                {'name':'control','configMap':{'name':'cpe-control','defaultMode':288}}]}}

        # Deployment maintains one application pod. Recreate avoids overlapping
        # application replicas while this demonstrator retains SQLite and one worker.
        self.apply({'apiVersion':'apps/v1','kind':'Deployment','metadata':{'name':'cpe-app','namespace':ns},'spec':{
            'replicas':1,'strategy':{'type':'Recreate'},'selector':{'matchLabels':{'app':'cpe-app'}},'template':pod}})

        # Service gives matching pods a stable internal address on port 8770.
        # No NodePort, LoadBalancer or OpenShift Route is created here. A Service
        # address is connectivity, not authorization to read the application's data.
        self.apply({'apiVersion':'v1','kind':'Service','metadata':{'name':'cpe-app','namespace':ns},'spec':{
            'selector':{'app':'cpe-app'},'ports':[{'port':8770,'targetPort':8770}]}})
        self.oc('rollout','status','deployment/cpe-app','-n',ns,'--timeout=240s')
        self.save('profile-'+key+'.json',profile)
        print('Provisioned inactive synthetic CPE '+key+'. Private credentials retained; none displayed.')

    def admission(self,ns,image):

        # Before admitting a pod, the API server evaluates these Common Expression
        # Language (CEL) conditions. They reject host access, unapproved containers,
        # mounted API tokens and unexpected volumes. Deny-on-error is requested.
        # These checks complement, rather than replace, network and case rights.
        # Namespace-scoped binding keeps all unrelated projects unaffected.
        validations=[
          "!has(object.spec.hostNetwork) || !object.spec.hostNetwork",
          "!has(object.spec.hostPID) || !object.spec.hostPID",
          "!has(object.spec.hostIPC) || !object.spec.hostIPC",
          "has(object.spec.automountServiceAccountToken) && object.spec.automountServiceAccountToken == false",
          "object.spec.serviceAccountName == 'cpe-runtime'",
          "!has(object.spec.initContainers) || size(object.spec.initContainers) == 0",
          "!has(object.spec.ephemeralContainers) || size(object.spec.ephemeralContainers) == 0",
          "size(object.spec.containers) == 1 && object.spec.containers.all(c, c.image == '"+image+"' && has(c.securityContext) && c.securityContext.readOnlyRootFilesystem == true)",
          "object.spec.volumes.all(v, (has(v.persistentVolumeClaim) && v.persistentVolumeClaim.claimName == 'cpe-data') || (has(v.secret) && v.secret.secretName in ['cpe-credentials','cpe-vector-client','cpe-inference-client']) || (has(v.configMap) && v.configMap.name == 'cpe-control') || (has(v.emptyDir) && v.name == 'tmp'))"
        ]
        name=ns+'-pod-contract'
        self.apply({'apiVersion':'admissionregistration.k8s.io/v1','kind':'ValidatingAdmissionPolicy','metadata':{'name':name},'spec':{
            'failurePolicy':'Fail','matchConstraints':{'resourceRules':[{'apiGroups':[''],'apiVersions':['v1'],
            'operations':['CREATE','UPDATE'],'resources':['pods']}]},
            'validations':[{'expression':x,'message':'Synthetic CPE pod contract violation'} for x in validations]}})
        self.apply({'apiVersion':'admissionregistration.k8s.io/v1','kind':'ValidatingAdmissionPolicyBinding','metadata':{'name':name},'spec':{
            'policyName':name,'validationActions':['Deny'],'matchResources':{'namespaceSelector':{'matchLabels':{'kubernetes.io/metadata.name':ns}}}}})

    def upgrade_image(self,key):
        ns=self.namespace(key)
        if self.get('namespace',ns)['metadata'].get('labels',{}).get('lab.multimodal.local/cpe')!='true':
            raise RuntimeError('Only operator-owned local CPEs can be upgraded')
        image=self.get('istag','lab-app:latest','multimodal-ai-lab')['image']['dockerImageReference']
        if '@sha256:' not in image:raise ValueError('An immutable local image is required')
        self.gate(key,'revoked')
        cm=self.get('configmap','cpe-profile',ns);profile=json.loads(cm['data']['profile.json'])
        profile['image']=image;cm['data']['profile.json']=json.dumps(profile);self.apply(cm)
        self.admission(ns,image)
        self.oc('set','image','deployment/cpe-app','app='+image,'-n',ns)
        self.oc('rollout','status','deployment/cpe-app','-n',ns,'--timeout=240s')
        self.save('profile-'+key+'.json',profile);self.gate(key,'provisioned')
        print('Updated pinned local CPE image. Fresh verification is mandatory before activation.')

    def renew(self,key,approval,hours=24):
        """Owner-requested renewal, retaining scope and data, never auto-activating."""

        # Renewal retains the existing project controls and data. It requires a
        # distinct owner approval, revokes the gate before changing expiry, and
        # finishes inactive. Old test evidence cannot activate the updated profile.
        if not isinstance(approval,str) or not approval.strip():
            raise ValueError('A new explicit owner approval reference is required')
        if type(hours) is not int or not 1<=hours<=24:
            raise ValueError('Renewal must be between one and twenty-four hours')
        ns=self.namespace(key)
        if self.get('namespace',ns)['metadata'].get('labels',{}).get('lab.multimodal.local/cpe')!='true':
            raise RuntimeError('Only operator-owned local CPEs can be renewed')
        cm=self.get('configmap','cpe-profile',ns)
        profile=json.loads(cm['data']['profile.json'])
        if profile['key']!=key or profile['namespace']!=ns or '@sha256:' not in profile['image']:
            raise RuntimeError('Renewal requires the unchanged pinned project profile')
        if approval.strip()==profile['approval']:
            raise ValueError('Renewal requires a distinct new approval reference')
        before=self.snapshot(key)
        self.save('renewal-'+key+'-'+str(time.time_ns())+'.json',
                  {'previous_profile':profile,'resource_digest':digest(before),'requested_hours':hours,
                   'new_approval':approval.strip(),'requested_at':time.time()})
        self.gate(key,'revoked')
        updated={**profile,'approval':approval.strip(),'expires':time.time()+hours*3600}

        # Optimistic concurrency: never overwrite a concurrently edited approval or scope.
        self.oc('replace','-f','-',payload={'apiVersion':'v1','kind':'ConfigMap',
            'metadata':{'name':'cpe-profile','namespace':ns,'resourceVersion':cm['metadata']['resourceVersion']},
            'data':{**cm['data'],'profile.json':json.dumps(updated)}})
        self.save('profile-'+key+'.json',updated)
        self.gate(key,'provisioned')
        print('Renewed local CPE '+key+' approval. Existing data and scope retained; fresh verify/activate required.')

    def snapshot(self,key):

        # Bind evidence to the resources actually configured, not just their names.
        # Include object identity, security settings and data fingerprints. Secret
        # values are hashed in this snapshot instead of copied into the report.
        ns=self.namespace(key);out={}
        for kind,name in [('namespace',ns),('resourcequota','cpe-budget'),('limitrange','cpe-container-limits'),
                          ('serviceaccount','cpe-runtime'),('networkpolicy','cpe-default-deny'),('networkpolicy','cpe-local-http'),
                          ('configmap','cpe-profile'),('secret','cpe-credentials'),('pvc','cpe-data'),('deployment','cpe-app'),('service','cpe-app')]:
            obj=self.get(kind,name,None if kind=='namespace' else ns)
            meta=obj['metadata'];out[kind+'/'+name]={'uid':meta['uid'],'spec':obj.get('spec'),
                'data_sha256':digest(obj.get('data',{})), 'labels':meta.get('labels',{}),
                'automount':obj.get('automountServiceAccountToken')}
        for kind in ['validatingadmissionpolicy','validatingadmissionpolicybinding']:
            obj=self.get(kind,ns+'-pod-contract');out[kind]={'uid':obj['metadata']['uid'],'spec':obj['spec']}
        obj=self.get('rolebinding',ns+'-image-pull','multimodal-ai-lab')
        out['image_pull_binding']={'uid':obj['metadata']['uid'],'subjects':obj['subjects'],'roleRef':obj['roleRef']}
        profile=json.loads(self.get('configmap','cpe-profile',ns)['data']['profile.json'])
        if profile.get('models'):
            for kind,name,target in [('secret','cpe-vector-client',ns),('secret','cpe-inference-client',ns),
                                      ('networkpolicy','cpe-rag-egress',ns),('networkpolicy','cpe-vector-ingress','multimodal-ai-lab'),
                                      ('networkpolicy','cpe-gateway-isolation','multimodal-ai-lab'),
                                      ('networkpolicy','lab-inference-isolation','multimodal-ai-lab'),
                                      ('deployment','lab-inference-gateway','multimodal-ai-lab'),
                                      ('secret','lab-inference-gateway','multimodal-ai-lab')]:
                obj=self.get(kind,name,target)
                out[target+'/'+kind+'/'+name]={'uid':obj['metadata']['uid'],'spec':obj.get('spec'),
                                              'data_sha256':digest(obj.get('data',{}))}
        return out

    def activate(self,key):

        # ACTIVATION DECISION: all recorded checks must pass, match the current
        # profile/resources, be at most ten minutes old, and precede expiry.
        # Any mismatch revokes access. Passing this local gate is not production
        # acceptance, nor permission for additional models or other projects.
        path=self.private/('verified-'+key+'.json');report=json.loads(path.read_text(encoding='utf-8'))
        profile=json.loads(self.get('configmap','cpe-profile',self.namespace(key))['data']['profile.json'])
        if (not report.get('passed') or not report.get('checks') or not all(x['passed'] for x in report['checks'])
                or report.get('profile_digest')!=digest(profile) or report.get('resource_digest')!=digest(self.snapshot(key))
                or time.time()-report['verified_at']>600 or profile['expires']<=time.time()):
            self.gate(key,'revoked');raise RuntimeError('Fresh passing evidence and unchanged controls are required')
        evidence=hashlib.sha256(path.read_bytes()).hexdigest()
        self.gate(key,'active',evidence)
        print('Activated local synthetic CPE '+key+' with verified resource/evidence binding.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['provision','activate','revoke','upgrade-image','renew'])
    parser.add_argument('--project',choices=PROJECTS,required=True)
    parser.add_argument('--admin-kubeconfig',type=Path,required=True)
    parser.add_argument('--private-dir',type=Path,required=True)
    parser.add_argument('--approval-reference')
    parser.add_argument('--hours',type=int,default=24)
    args=parser.parse_args();controller=Controller(args.admin_kubeconfig,args.private_dir)
    if args.command=='provision':controller.provision(args.project,args.approval_reference or '')
    elif args.command=='activate':controller.activate(args.project)
    elif args.command=='upgrade-image':controller.upgrade_image(args.project)
    elif args.command=='renew':controller.renew(args.project,args.approval_reference or '',args.hours)
    else:controller.gate(args.project,'revoked');print('CPE revoked. Volumes and audit evidence retained.')


if __name__=='__main__':main()
