"""Opt-in offline RAG extension for the two owned local CPEs; never activates without proof."""
import argparse
import base64
import copy
import json
import secrets
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.local_cpe import Controller
from lab.inference_gateway import MODELS


def extend(c):

    # --- BEFORE CHANGING THE SIX CONTROLS ---
    # c is the operator's local Controller. main is the namespace of the shared
    # model runtime, vector server and gateway. Project applications remain in
    # lab-cpe-a/b. Sharing a model server does not merge their document rights.
    main='multimodal-ai-lab'
    image=c.get('istag','lab-app:latest',main)['image']['dockerImageReference']
    if '@sha256:' not in image:raise RuntimeError('Immutable image required')

    # Revoke both before changing any protected shared control.
    for key in ['a','b']:c.gate(key,'revoked')

    # --- CONTROL 4: SEPARATE PROJECT CREDENTIALS ---
    # Reuse existing gateway tokens or generate one per project. A token identifies
    # project A or B to the gateway, not the human Officer using the application.
    # stringData lets Kubernetes encode the supplied strings into Secret data.
    # It does not encrypt them or make it safe to log or commit the object.
    result=c.oc('get','secret','lab-inference-gateway','-n',main,'--ignore-not-found','-o','json').stdout.strip()
    tokens=json.loads(base64.b64decode(json.loads(result)['data']['tokens.json'])) if result else {key:secrets.token_urlsafe(36) for key in ['a','b']}
    c.apply({'apiVersion':'v1','kind':'Secret','metadata':{'name':'lab-inference-gateway','namespace':main},
             'stringData':{'tokens.json':json.dumps(tokens)}})

    # --- CONTROL 6: INTERNAL MODEL ACCESS POINT ---
    # This internal Service listens on 8771 and forwards to gateway pods selected
    # by their app label. It creates neither an Internet route nor a public API.
    c.apply({'apiVersion':'v1','kind':'Service','metadata':{'name':'lab-inference-gateway','namespace':main},
             'spec':{'selector':{'app':'lab-inference-gateway'},'ports':[{'port':8771,'targetPort':8771}]}})

    # --- RUN THE GATEWAY APPLICATION ---
    # The Deployment runs one FastAPI gateway using the pinned application image.
    # The gateway reads tokens from a read-only Secret, runs without an API token,
    # and enforces model/operation/request limits in lab/inference_gateway.py.
    # One replica matters: its request counters and inference slot are in memory.
    c.apply({'apiVersion':'apps/v1','kind':'Deployment','metadata':{'name':'lab-inference-gateway','namespace':main},
      'spec':{'replicas':1,'selector':{'matchLabels':{'app':'lab-inference-gateway'}},'template':{
        'metadata':{'labels':{'app':'lab-inference-gateway'}},'spec':{'automountServiceAccountToken':False,
        'enableServiceLinks':False,'securityContext':{'runAsNonRoot':True,'seccompProfile':{'type':'RuntimeDefault'}},
        'containers':[{'name':'gateway','image':image,'command':['python','-m','uvicorn','lab.inference_gateway:create_gateway','--factory','--host','0.0.0.0','--port','8771','--no-access-log'],
          'env':[{'name':'LAB_GATEWAY_TOKENS_FILE','value':'/secrets/tokens.json'}],
          'resources':{'requests':{'cpu':'50m','memory':'128Mi'},'limits':{'cpu':'500m','memory':'256Mi'}},
          'securityContext':{'allowPrivilegeEscalation':False,'readOnlyRootFilesystem':True,'capabilities':{'drop':['ALL']}},
          'readinessProbe':{'httpGet':{'path':'/health/ready','port':8771}},
          'volumeMounts':[{'name':'credentials','mountPath':'/secrets','readOnly':True}]}],
        'volumes':[{'name':'credentials','secret':{'secretName':'lab-inference-gateway','defaultMode':288}}]}}}})

    # --- CONTROL 2: WHICH PODS MAY CONTACT THE GATEWAY ---
    # In EACH peer, namespaceSelector AND podSelector must both match. This means
    # cpe-app pods in the named project namespaces, not every pod in the cluster.
    # The gateway's egress allows only local Ollama on 11434 and cluster DNS.
    # Network reachability still requires the matching project token at FastAPI.
    cpes=[{'namespaceSelector':{'matchLabels':{'kubernetes.io/metadata.name':'lab-cpe-'+key}},
           'podSelector':{'matchLabels':{'app':'cpe-app'}}} for key in ['a','b']]
    c.apply({'apiVersion':'networking.k8s.io/v1','kind':'NetworkPolicy','metadata':{'name':'cpe-gateway-isolation','namespace':main},
      'spec':{'podSelector':{'matchLabels':{'app':'lab-inference-gateway'}},'policyTypes':['Ingress','Egress'],
       'ingress':[{'from':cpes,'ports':[{'port':8771}]}],
       'egress':[{'to':[{'podSelector':{'matchLabels':{'app.kubernetes.io/name':'lab-inference'}}}],'ports':[{'port':11434}]},
                 {'to':[{'namespaceSelector':{'matchLabels':{'kubernetes.io/metadata.name':'openshift-dns'}},
                         'podSelector':{'matchLabels':{'dns.operator.openshift.io/daemonset-dns':'default'}}}],
                  'ports':[{'port':5353,'protocol':'UDP'},{'port':5353,'protocol':'TCP'}]}]}})

    # --- PREVENT DIRECT MODEL BYPASS ---
    # Add the gateway as an allowed caller on Ollama's OWN ingress policy. Project
    # applications are not added as direct callers. Both source and destination
    # network policies must permit the intended connection.
    policy=c.get('networkpolicy','lab-inference-isolation',main)
    policy['spec']['ingress'].append({'from':[{'podSelector':{'matchLabels':{'app':'lab-inference-gateway'}}}], 'ports':[{'port':11434,'protocol':'TCP'}]})

    # Deduplicate on subsequent runs.
    policy['spec']['ingress']=list({json.dumps(x,sort_keys=True):x for x in policy['spec']['ingress']}.values());c.apply(policy)

    # --- CONTROL 5: DATABASE CONNECTIONS ---
    # Project pods may reach the shared PostgreSQL server on 5432. The network
    # policy does not distinguish its databases. The per-project SQL roles and
    # CONNECT privileges below enforce that second, separate storage boundary.
    c.apply({'apiVersion':'networking.k8s.io/v1','kind':'NetworkPolicy','metadata':{'name':'cpe-vector-ingress','namespace':main},
      'spec':{'podSelector':{'matchLabels':{'app.kubernetes.io/name':'lab-vector'}},'policyTypes':['Ingress'],
              'ingress':[{'from':cpes,'ports':[{'port':5432}]}]}})
    vector_ip=c.get('service','lab-vector',main)['spec']['clusterIP']
    gateway_ip=c.get('service','lab-inference-gateway',main)['spec']['clusterIP']
    config=c.get('configmap','lab-rag-config',main)['data']
    c.oc('rollout','status','deployment/lab-inference-gateway','-n',main,'--timeout=240s')
    for key in ['a','b']:
        ns=c.namespace(key)
        profile=json.loads(c.get('configmap','cpe-profile',ns)['data']['profile.json'])
        role='cpe_vector_'+key;database='cpe_vectors_'+key
        old=c.oc('get','secret','cpe-vector-client','-n',ns,'--ignore-not-found','-o','json').stdout.strip()
        password=base64.b64decode(json.loads(old)['data']['password']).decode() if old else secrets.token_urlsafe(36)

        # Values are generated URL-safe and SQL uses stdin, never command arguments/output.
        if not all(ch.isalnum() or ch in '-_' for ch in password):raise ValueError('Unexpected generated credential')

        # --- CONTROL 5: A DATABASE AND NON-ADMIN ROLE PER PROJECT ---
        # Remove superuser/database-creation/role-creation powers. Revoke the
        # default PUBLIC connection permission, then grant this role access to
        # its own database. The retrieval schema belongs to that project's role.
        # Application case/document authorization is still required within it.
        # Credential-bearing SQL goes through stdin and is never printed here.
        sql=rf"""DO $$ BEGIN IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname='{role}') THEN CREATE ROLE {role} LOGIN; END IF; END $$;
ALTER ROLE {role} NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION PASSWORD '{password}';
SELECT 'CREATE DATABASE {database} OWNER vector_admin' WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname='{database}')\gexec
REVOKE CONNECT ON DATABASE {database} FROM PUBLIC;
REVOKE CONNECT ON DATABASE lab_vectors,postgres,template1 FROM PUBLIC;
GRANT CONNECT ON DATABASE {database} TO {role};
\connect {database}
CREATE EXTENSION IF NOT EXISTS vector;
REVOKE CREATE ON SCHEMA public FROM PUBLIC;
CREATE SCHEMA IF NOT EXISTS retrieval AUTHORIZATION {role};
"""
        import subprocess
        result=subprocess.run(['oc','--kubeconfig',c.kubeconfig,'exec','-i','-n',main,'lab-vector-0','--','psql','-v','ON_ERROR_STOP=1','-U','vector_admin','-d','lab_vectors'],input=sql,capture_output=True,text=True,encoding='utf-8',timeout=60)
        if result.returncode:raise RuntimeError('Isolated vector database provisioning failed; credential-bearing SQL withheld')

        # --- CONTROL 4: GIVE EACH PROJECT ONLY ITS OWN SERVICE SECRETS ---
        # The application receives its database password and gateway token in its
        # own namespace. It does not receive the PostgreSQL administrator password
        # or the gateway's map of all project tokens. Mounts below are read-only.
        for name,data in [('cpe-vector-client',{'password':password}),('cpe-inference-client',{'token':tokens[key]})]:
            c.apply({'apiVersion':'v1','kind':'Secret','metadata':{'name':name,'namespace':ns},'stringData':data})

        # --- CONTROL 2: THREE APPROVED OUTBOUND CONNECTIONS ---
        # Allow this project's app to contact the gateway, vector server and DNS.
        # These allow rules complement the baseline default-deny policy. They do
        # not authorize Internet traffic, another project's app or direct Ollama.
        c.apply({'apiVersion':'networking.k8s.io/v1','kind':'NetworkPolicy','metadata':{'name':'cpe-rag-egress','namespace':ns},
          'spec':{'podSelector':{'matchLabels':{'app':'cpe-app'}},'policyTypes':['Egress'],
           'egress':[{'to':[{'namespaceSelector':{'matchLabels':{'kubernetes.io/metadata.name':main}},
                            'podSelector':{'matchLabels':{'app': 'lab-inference-gateway'}}}], 'ports':[{'port':8771}]},
                     {'to':[{'namespaceSelector':{'matchLabels':{'kubernetes.io/metadata.name':main}},
                             'podSelector':{'matchLabels':{'app.kubernetes.io/name':'lab-vector'}}}], 'ports':[{'port':5432}]},
                     {'to':[{'namespaceSelector':{'matchLabels':{'kubernetes.io/metadata.name':'openshift-dns'}},
                             'podSelector':{'matchLabels':{'dns.operator.openshift.io/daemonset-dns':'default'}}}],
                      'ports':[{'port':5353,'protocol':'UDP'},{'port':5353,'protocol':'TCP'}]}]}})

        # --- CONTROL 6: EXACT MODELS, NOT AN UNRESTRICTED AI ENDPOINT ---
        # Record the permitted model names AND content digests. The gateway uses
        # BGE-M3 for embeddings and Qwen3:4b for generation. No extra model is
        # approved simply because it is installed on the shared server.
        profile.update(image=image,models=[{'name':name,'digest':value} for name,value in MODELS.items()],
                       scope='synthetic_documents_offline_rag',vector_database=database,vector_user=role,
                       inference_gateway='lab-inference-gateway.multimodal-ai-lab.svc:8771')
        c.apply({'apiVersion':'v1','kind':'ConfigMap','metadata':{'name':'cpe-profile','namespace':ns},'data':{'profile.json':json.dumps(profile)}})
        c.admission(ns,image)
        deploy=c.get('deployment','cpe-app',ns)
        pod=deploy['spec']['template']['spec'];container=pod['containers'][0];container['image']=image

        # --- CONNECT THE EXISTING PROJECT APPLICATION ---
        # Point retrieval at the project's database/role and generation at the
        # gateway. Service addresses are not credentials: read-only Secret files
        # supply the matching credentials without putting them in these settings.
        settings={**config,'LAB_VECTOR_HOST':vector_ip,'LAB_VECTOR_DATABASE':database,'LAB_VECTOR_USER':role,
                  'LAB_INFERENCE_URL':'http://lab-inference-gateway.multimodal-ai-lab.svc:8771',
                  'LAB_INFERENCE_TOKEN_FILE':'/app/secrets/inference/token'}
        env={x['name']:x for x in container['env']}
        env.update({k:{'name':k,'value':v} for k,v in settings.items()});container['env']=list(env.values())
        for volume,secret,mount in [('vector-password','cpe-vector-client','/app/secrets/vector'),('inference-token','cpe-inference-client','/app/secrets/inference')]:
            container['volumeMounts']=[v for v in container['volumeMounts'] if v['name']!=volume]+[{'name':volume,'mountPath':mount,'readOnly':True}]
            pod['volumes']=[v for v in pod['volumes'] if v['name']!=volume]+[{'name':volume,'secret':{'secretName':secret,'defaultMode':288}}]
        c.apply(deploy)
        c.oc('rollout','status','deployment/cpe-app','-n',ns,'--timeout=240s')

        # --- VERIFY BEFORE ACTIVATION ---
        # Provisioned means configured but inactive. Run the independent RAG and
        # isolation checks, then ask Controller.activate to bind passing evidence
        # to the current profile/resources. This script does not activate a CPE.
        c.gate(key,'provisioned');c.save('profile-'+key+'.json',profile)
        print('Extended local CPE '+key+'; inactive pending actual RAG and isolation verification.',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--admin-kubeconfig',type=Path,required=True);p.add_argument('--private-dir',type=Path,required=True)
    args=p.parse_args();extend(Controller(args.admin_kubeconfig,args.private_dir))

if __name__=='__main__':main()
