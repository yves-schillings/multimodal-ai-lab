"""Recurring synthetic classifier canary. No case text or production drift claim."""
import hashlib
import json
import pickle
import threading
import time
import uuid

from .store import _now

CANARY={
    "witness_statement":["Witness statement describing an observed vehicle and testimony.","Interview testimony from a witness who heard a sound."],
    "incident_report":["Patrol incident report: officer attendance and response at the scene.","Police response incident log for damage to property."],
    "evidence_inventory":["Evidence inventory listing exhibits and their custody identifiers.","Seized evidence item manifest: collected photographs and recordings."],
    "correspondence":["Dear colleague, please reply to this correspondence email. Regards.","Letter correspondence inviting colleagues to an appointment."],
}


class QualityMonitor:
    def __init__(self,registry,interval=1800):
        if not 60<=interval<=86400:
            raise ValueError("Monitoring interval must be 60 to 86400 seconds.")
        self.registry,self.store,self.interval=registry,registry.store,interval
        self.stop_event,self.lock=threading.Event(),threading.Lock()
        self.thread=None
        with self.store._connection(write=True) as db:
            db.execute("CREATE TABLE IF NOT EXISTS model_monitor_runs(id TEXT PRIMARY KEY, created_at TEXT NOT NULL, result_json TEXT NOT NULL)")

    def state(self):
        with self.store._connection() as db:
            runs=[json.loads(row['result_json']) for row in db.execute("SELECT result_json FROM model_monitor_runs ORDER BY created_at DESC LIMIT 20")]
        return {"enabled":True,"interval_seconds":self.interval,"runs":runs,
                "scope":"Fixed synthetic classifier canary; not live-data drift monitoring. No automatic promotion or rollback."}

    def run(self):
        if not self.lock.acquire(blocking=False):
            raise ValueError("Classifier monitoring is already running.")
        report={"id":uuid.uuid4().hex,"created_at":_now(),"status":"unavailable","data_scope":"synthetic_canary_only",
                "model_version":None,"mlflow_run_id":None,"notice":"Small synthetic evaluation; production quality and real-data drift remain unverified."}
        client,run_id=None,None
        try:
            state=self.registry.state();version=state['active_version']
            if not version:
                report['reason']='No promoted classifier is available.'
                return report
            report['model_version']=version
            candidate=next(v for v in state['versions'] if v['version']==version)
            self.registry._verify_release(candidate)
            data=(self.registry.root/(version+'.pkl')).read_bytes()
            if hashlib.sha256(data).hexdigest()!=candidate['artifact_sha256']:
                raise ValueError('Changed classifier artifact.')
            model=pickle.loads(data)  # Checksum-verified artifact produced by this local lab.
            examples,labels=[],[]
            for label,texts in CANARY.items():
                examples.extend(texts);labels.extend([label]*len(texts))
            from sklearn.metrics import accuracy_score,f1_score
            start=time.monotonic();predicted=model.predict(examples)
            metrics={'accuracy':float(accuracy_score(labels,predicted)), 'macro_f1':float(f1_score(labels,predicted,average='macro')),
                     'sample_count':len(examples),'prediction_seconds':time.monotonic()-start}
            report['metrics']=metrics
            if self.registry.state()['active_version']!=version:
                report.update(status='stale',reason='Release changed during evaluation.')
                return report
            client=self.registry._client(candidate)
            experiment=client.get_experiment_by_name('synthetic-classifier-monitoring')
            eid=experiment.experiment_id if experiment else client.create_experiment('synthetic-classifier-monitoring')
            run_id=client.create_run(eid,tags={'data_scope':'synthetic_canary_only','mlflow.runName':'classifier-canary','model_version':version}).info.run_id
            report['mlflow_run_id']=run_id
            for name,value in metrics.items():client.log_metric(run_id,name,float(value))
            client.log_param(run_id,'canary_sha256',hashlib.sha256(json.dumps(CANARY,sort_keys=True).encode()).hexdigest())
            client.log_param(run_id,'artifact_sha256',candidate['artifact_sha256'])
            report['status']='healthy' if metrics['accuracy']>=0.85 and metrics['macro_f1']>=0.85 else 'alert'
            client.set_tag(run_id,'canary_status',report['status']);client.set_terminated(run_id,'FINISHED')
            return report
        except Exception:
            report.update(status='unavailable',reason='Model provenance, evaluation or MLflow recording could not be verified.')
            if client and run_id:
                try:client.set_terminated(run_id,'FAILED')
                except Exception:pass
            return report
        finally:
            try:
                with self.store._connection(write=True) as db:
                    db.execute('INSERT INTO model_monitor_runs VALUES(?,?,?)',(report['id'],report['created_at'],json.dumps(report)))
            finally:self.lock.release()

    def start(self):
        if self.thread:return
        def loop():
            while not self.stop_event.wait(self.interval):
                try:self.run()
                except Exception:continue  # A failed evaluation cannot change a model release.
        self.thread=threading.Thread(target=loop,name='classifier-canary-monitor',daemon=True)
        self.thread.start()

    def close(self):
        self.stop_event.set()
        if self.thread:self.thread.join(timeout=5)
