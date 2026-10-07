"""Real local sklearn training on separated synthetic splits; explicit quality gate."""
import importlib.util
import hashlib
import json
import os
import pickle
import threading
import tempfile
import uuid
from pathlib import Path
from .store import _now

# Training examples and held-out templates are distinct. Scores measure this tiny
# synthetic exercise only; they do not establish multilingual police suitability.
TRAIN = {
 'witness_statement': ['Witness statement: I saw the bicycle near the library.', 'Interview testimony: the witness heard a noise.', 'Statement by a person describing what they observed.', 'Witness reports seeing a man walking yesterday.', 'Interview record of personal observations and testimony.', 'Testimony: I noticed a damaged window in the morning.'],
 'incident_report': ['Incident report: officers attended a theft scene.', 'Patrol report describing an incident and police response.', 'Incident log: damage reported to a building.', 'Officer report: response to stolen property incident.', 'Police incident report with date location and response.', 'Patrol incident log records attendance at a burglary.'],
 'evidence_inventory': ['Evidence inventory: item one photograph, item two recording.', 'Inventory of seized evidence items and custody references.', 'Evidence exhibit list: camera, bag, fingerprints.', 'Evidence register lists item identifiers and custody.', 'Inventory: collected physical evidence exhibits.', 'Evidence item manifest: photographs recordings and sealed packages.'],
 'correspondence': ['Correspondence email: Dear colleague, please attend the meeting.', 'Email message: Hello, thanks for your reply. Regards.', 'Letter correspondence requesting an appointment.', 'Dear team, this email confirms the meeting schedule.', 'Correspondence: please reply to this letter. Kind regards.', 'Email invitation to a scheduled discussion with colleagues.']}
TEST = {
 'witness_statement': ['Witness testimony: I observed a blue vehicle.', 'Interview statement describing what a witness saw.'],
 'incident_report': ['Incident report: patrol response to property damage.', 'Police officers attended the scene in this incident log.'],
 'evidence_inventory': ['Evidence inventory of collected recording exhibits.', 'Seized item custody register and evidence manifest.'],
 'correspondence': ['Dear colleagues, please reply to this email invitation.', 'Letter correspondence confirming an appointment.']}


class ModelRegistry:
    registered_name = 'multimodal-document-classifier'

    def __init__(self,store):
        self.store=store
        self.root=store.root/'models'
        self.root.mkdir(exist_ok=True)
        self.lock=threading.RLock()

    def state(self):
        with self.store._connection() as db:
            versions=[json.loads(row['metadata_json']) for row in db.execute('SELECT * FROM model_versions ORDER BY created_at')]
            row=db.execute("SELECT value FROM settings WHERE key='model_state'").fetchone()
        state=json.loads(row['value']) if row else {'active_version':None,'previous_version':None}
        return {**state,'versions':versions,'available':importlib.util.find_spec('sklearn') is not None,
                'task':'synthetic_document_classification','quality_gate':{'minimum_accuracy':0.85,'minimum_macro_f1':0.85},
                'notice':'Small English synthetic hold-out evaluation only; not production validation.'}

    def train(self):
        from sklearn.pipeline import Pipeline
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score,f1_score
        x,y=[],[]
        for label,samples in TRAIN.items():
            x.extend(samples); y.extend([label]*len(samples))
        test_x,test_y=[],[]
        for label,samples in TEST.items():
            test_x.extend(samples); test_y.extend([label]*len(samples))
        model=Pipeline([('tfidf',TfidfVectorizer(ngram_range=(1,2))),('classifier',LogisticRegression(C=8,random_state=17,max_iter=500))])
        model.fit(x,y)
        predictions=model.predict(test_x)
        accuracy=float(accuracy_score(test_y,predictions)); macro=float(f1_score(test_y,predictions,average='macro'))
        version='classifier-'+uuid.uuid4().hex[:12]
        metadata={'version':version,'accuracy':accuracy,'macro_f1':macro,'gate_passed':accuracy>=0.85 and macro>=0.85,'training_samples':len(x),'test_samples':len(test_x),'data_scope':'synthetic_only','created_at':_now(),'mlflow_run_id':None,'mlflow_status':'unavailable'}
        model_file=self.root/(version+'.pkl')
        with model_file.open('wb') as file:
            pickle.dump(model,file)
        metadata['artifact_sha256']=hashlib.sha256(model_file.read_bytes()).hexdigest()
        tracking=self.store.root/'mlflow'; tracking.mkdir(exist_ok=True)
        # MLFLOW_TRACKING_URI selects a tracking server (Docker Compose / OpenShift);
        # without it, metrics stay in the local SQLite file under the runtime directory.
        server_uri=os.environ.get('MLFLOW_TRACKING_URI','').strip()
        metadata['mlflow_tracking_uri']=server_uri or 'sqlite:///'+(tracking/'mlflow.db').resolve().as_posix()
        run=None
        try:
            import mlflow.sklearn
            client=self._client(metadata)
            experiment=client.get_experiment_by_name('synthetic-document-classification')
            if experiment:
                experiment_id=experiment.experiment_id
            elif server_uri:
                experiment_id=client.create_experiment('synthetic-document-classification')
            else:
                experiment_id=client.create_experiment('synthetic-document-classification',artifact_location=tracking.resolve().as_uri())
            run=client.create_run(experiment_id,tags={'data_scope':'synthetic_only','mlflow.runName':version})
            for name in ('accuracy','macro_f1','training_samples','test_samples'):
                client.log_metric(run.info.run_id,name,float(metadata[name]))
            client.log_param(run.info.run_id,'algorithm','tfidf_logistic_regression')
            client.log_param(run.info.run_id,'gate_passed',str(metadata['gate_passed']))
            client.log_param(run.info.run_id,'artifact_sha256',metadata['artifact_sha256'])
            # Publish a loadable MLflow model plus the exact local inference artifact.
            # No recordings, case documents or source text enter this experiment.
            with tempfile.TemporaryDirectory() as directory:
                model_dir=Path(directory)/'model'
                mlflow.sklearn.save_model(model,str(model_dir),pip_requirements=['scikit-learn=='+__import__('sklearn').__version__])
                client.log_artifacts(run.info.run_id,str(model_dir),'model')
            client.log_artifact(run.info.run_id,str(model_file),'release')
            try:
                client.get_registered_model(self.registered_name)
            except Exception as error:
                if getattr(error,'error_code',None) != 'RESOURCE_DOES_NOT_EXIST':
                    raise
                client.create_registered_model(self.registered_name)
            registered=client.create_model_version(self.registered_name,
                'runs:/'+run.info.run_id+'/model',run_id=run.info.run_id,
                tags={'local_version':version,'data_scope':'synthetic_only','artifact_sha256':metadata['artifact_sha256']})
            if registered.status != 'READY':
                raise ValueError('MLflow model version is not ready.')
            client.set_terminated(run.info.run_id,'FINISHED')
            metadata.update(mlflow_run_id=run.info.run_id,mlflow_status='recorded_on_server' if server_uri else 'recorded_locally',
                mlflow_model_name=self.registered_name,mlflow_model_version=str(registered.version),
                mlflow_model_uri='models:/'+self.registered_name+'/'+str(registered.version),artifact_published=True)
        except Exception:
            metadata['mlflow_status']='server_logging_failed' if server_uri else 'local_logging_failed'
            metadata['artifact_published']=False
            if run is not None:
                try:
                    client.set_terminated(run.info.run_id,'FAILED')
                except Exception:
                    pass
        with self.lock:
            with self.store._connection(write=True) as db:
                db.execute('INSERT INTO model_versions VALUES(?,?,?)',(version,json.dumps(metadata),metadata['created_at']))
        return metadata

    def promote(self,actor,version=None):
        self.store._actor(actor)
        if actor!='reviewer':
            raise PermissionError('Only the simulated reviewer may promote models.')
        with self.lock:
            state=self.state()
            candidates=[v for v in state['versions'] if version is None or v['version']==version]
            if not candidates:
                raise ValueError('Train a candidate model before promotion.')
            candidate=candidates[-1]
            if not candidate['gate_passed']:
                raise ValueError('Candidate failed the held-out quality gate.')
            self._verify_release(candidate)
            if candidate['version']==state['active_version']:
                return state
            self._write_state({'active_version':candidate['version'],'previous_version':state['active_version']})
            return self.state()

    def rollback(self,actor):
        self.store._actor(actor)
        if actor!='reviewer':
            raise PermissionError('Only the simulated reviewer may roll back models.')
        with self.lock:
            state=self.state()
            if state['previous_version'] is None:
                raise ValueError('No previous promoted model is available.')
            candidate=next(v for v in state['versions'] if v['version']==state['previous_version'])
            self._verify_release(candidate)
            self._write_state({'active_version':state['previous_version'],'previous_version':state['active_version']})
            return self.state()

    def _client(self,metadata):
        from mlflow.tracking import MlflowClient
        return MlflowClient(tracking_uri=metadata['mlflow_tracking_uri'],registry_uri=metadata['mlflow_tracking_uri'])

    def _verify_release(self,candidate):
        if not candidate.get('artifact_published') or candidate.get('mlflow_status') not in {'recorded_locally','recorded_on_server'}:
            raise ValueError('Release requires successful MLflow artifact publication and registration.')
        local=self.root/(candidate['version']+'.pkl')
        if not local.is_file() or hashlib.sha256(local.read_bytes()).hexdigest()!=candidate['artifact_sha256']:
            raise ValueError('Local model artifact does not match its registered checksum.')
        try:
            client=self._client(candidate)
            registered=client.get_model_version(candidate['mlflow_model_name'],candidate['mlflow_model_version'])
            if registered.status!='READY' or registered.run_id!=candidate['mlflow_run_id'] or registered.tags.get('artifact_sha256')!=candidate['artifact_sha256']:
                raise ValueError('Registry identity or artifact provenance mismatch.')
            if client.get_run(candidate['mlflow_run_id']).info.status!='FINISHED':
                raise ValueError('The MLflow training run did not finish successfully.')
            with tempfile.TemporaryDirectory() as directory:
                published=Path(client.download_artifacts(candidate['mlflow_run_id'],'release/'+local.name,directory))
                if hashlib.sha256(published.read_bytes()).hexdigest()!=candidate['artifact_sha256']:
                    raise ValueError('Published model artifact checksum mismatch.')
        except Exception as error:
            raise ValueError('Release blocked: MLflow provenance and published artifact must be verifiable.') from error

    def _write_state(self,state):
        with self.store._connection(write=True) as db:
            db.execute("INSERT INTO settings(key,value) VALUES('model_state',?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",(json.dumps(state),))

    def classify(self,text):
        with self.lock:
            version=self.state()['active_version']
            if not version:
                return {'label':None,'status':'unavailable','reason':'Train and promote a local classifier first.'}
            with open(self.root/(version+'.pkl'),'rb') as file:
                model=pickle.load(file)
            probabilities=model.predict_proba([text[:300000]])[0]
            index=int(probabilities.argmax())
            return {'label':str(model.classes_[index]),'confidence':round(float(probabilities[index]),4),'model_version':version,'status':'classified','requires_human_validation':True}
