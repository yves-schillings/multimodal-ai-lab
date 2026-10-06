"""Single-process SQLite queue with durable recovery, not RabbitMQ."""
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from .documents import extract_document


class JobRunner:
    def __init__(self,store,registry,speech):
        self.store,self.registry,self.speech=store,registry,speech
        self.executor=ThreadPoolExecutor(max_workers=1,thread_name_prefix='local-lab')
        self.lock=threading.Lock()
        self.submitted=set()

    def recover(self):
        with self.store._connection(write=True) as db:
            db.execute("UPDATE jobs SET status='pending' WHERE status='running'")
            ids=[row['id'] for row in db.execute("SELECT id FROM jobs WHERE status='pending'")]
        for job_id in ids:
            self.submit(job_id)

    def submit(self,job_id):
        with self.lock:
            if job_id in self.submitted:
                return
            self.submitted.add(job_id)
        self.executor.submit(self._run,job_id)

    def _run(self,job_id):
        with self.store._connection() as db:
            job=dict(db.execute('SELECT * FROM jobs WHERE id=?',(job_id,)).fetchone())
        if job['status']!='pending':
            return
        self.store.job_update(job_id,'running')
        payload=json.loads(job['payload_json'])
        try:
            if job['kind']=='train':
                result=self.registry.train()
            else:
                self.store.get_case(job['actor'],job['case_id'])
                if job['kind']=='audio':
                    transcription=self.speech.transcribe(Path(payload['path']),payload['language'])
                    transcription['engine']='faster-whisper-local'
                    case=self.store.save_transcript_job(job['actor'],job['case_id'],transcription,job_id)
                    result={'case_id':case['id'],'case_revision':case['revision'],'transcription':transcription}
                else:
                    extraction=extract_document(payload['path'])
                    result=self.store.add_document(job['actor'],job['case_id'],payload['name'],extraction,self.registry.classify(extraction['text']),job_id)
            self.store.job_update(job_id,'completed',result=result)
        except (ValueError,RuntimeError) as exc:
            # These expected local adapters provide fixed, content-free errors.
            self.store.job_update(job_id,'failed',error=str(exc)[:250])
        except Exception:
            self.store.job_update(job_id,'failed',error='Local processing failed. Check dependency/model availability or file format.')

    def close(self):
        self.executor.shutdown(wait=True,cancel_futures=False)
