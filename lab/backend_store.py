"""Durable local documents, jobs and model registry; case access checked first."""
import json
import uuid
from .store import StatementStore, _now, _hash, _stamp


class LabStore(StatementStore):
    def __init__(self, root):
        super().__init__(root)
        with self._connection(write=True) as db:
            db.executescript('''
            CREATE TABLE IF NOT EXISTS documents(id TEXT PRIMARY KEY, case_id TEXT NOT NULL REFERENCES cases(id), name TEXT NOT NULL, original_text TEXT NOT NULL, text TEXT NOT NULL, engine TEXT NOT NULL, classification_json TEXT NOT NULL, reviewed INTEGER NOT NULL DEFAULT 0, reviewed_by TEXT, created_at TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS jobs(id TEXT PRIMARY KEY, actor TEXT NOT NULL, case_id TEXT, kind TEXT NOT NULL, status TEXT NOT NULL, payload_json TEXT NOT NULL, result_json TEXT, error TEXT, digest TEXT NOT NULL UNIQUE, created_at TEXT NOT NULL, updated_at TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS model_versions(version TEXT PRIMARY KEY, metadata_json TEXT NOT NULL, created_at TEXT NOT NULL);
            ''')

    def get_case(self, actor, case_id):
        case = super().get_case(actor, case_id)
        with self._connection() as db:
            self._case(db, actor, case_id)
            case['documents'] = []
            for row in db.execute('SELECT * FROM documents WHERE case_id=? ORDER BY created_at', (case_id,)):
                item = dict(row)
                item['classification'] = json.loads(item.pop('classification_json'))
                item['reviewed'] = bool(item['reviewed'])
                case['documents'].append(item)
        return case

    def add_document(self, actor, case_id, name, extraction, classification, job_id=None):
        document_id = 'document-' + uuid.uuid4().hex
        with self._connection(write=True) as db:
            case = self._case(db, actor, case_id)
            key = 'ingest:' + job_id if job_id else None
            saved = db.execute('SELECT value FROM settings WHERE key=?', (key,)).fetchone() if key else None
            if saved:
                return json.loads(saved['value'])
            revision = case['revision'] + 1
            db.execute('INSERT INTO documents(id,case_id,name,original_text,text,engine,classification_json,created_at) VALUES(?,?,?,?,?,?,?,?)',
                       (document_id, case_id, name, extraction['text'], extraction['text'], extraction['engine'], json.dumps(classification), _now()))
            db.execute("UPDATE cases SET revision=?,draft_json=NULL,approved_revision=NULL,approved_by=NULL,status='in_review',updated_at=? WHERE id=?", (revision, _now(), case_id))
            self._audit(db, case_id, actor, 'document_extracted', revision, {'document_id':document_id,'content_hash':_hash(extraction['text']), 'engine':extraction['engine']})
            result = {'document_id': document_id, 'classification': classification, 'case_revision':revision}
            if key:
                db.execute('INSERT INTO settings(key,value) VALUES(?,?)',(key,json.dumps(result)))
        return result

    def save_transcript_job(self, actor, case_id, transcription, job_id):
        with self._connection(write=True) as db:
            self._case(db, actor, case_id)
            key = 'ingest:' + job_id
            saved = db.execute('SELECT value FROM settings WHERE key=?', (key,)).fetchone()
            if not saved:
                self._save_transcript(db, actor, case_id, transcription)
                db.execute('INSERT INTO settings(key,value) VALUES(?,?)',(key,'{}'))
        return self.get_case(actor, case_id)

    def review_document(self, actor, case_id, document_id, text, expected_revision):
        with self._connection(write=True) as db:
            case = self._case(db, actor, case_id)
            self._revision(case, expected_revision)
            doc = db.execute('SELECT * FROM documents WHERE id=? AND case_id=?', (document_id,case_id)).fetchone()
            if doc is None:
                raise KeyError('Document not found in this case.')
            value = doc['text'] if text is None else text
            if not isinstance(value,str) or not value.strip() or len(value)>300000:
                raise ValueError('Reviewed document requires 1 to 300000 characters.')
            revision = case['revision']+1
            db.execute('UPDATE documents SET text=?,reviewed=1,reviewed_by=? WHERE id=?', (value.strip(),actor,document_id))
            db.execute("UPDATE cases SET revision=?,draft_json=NULL,approved_revision=NULL,approved_by=NULL,status='in_review',updated_at=? WHERE id=?", (revision,_now(),case_id))
            self._audit(db,case_id,actor,'document_reviewed',revision,{'document_id':document_id,'content_hash':_hash(value)})
        return self.get_case(actor,case_id)

    def create_job(self, actor, case_id, kind, payload, digest):
        self._actor(actor)
        with self._connection(write=True) as db:
            if case_id:
                self._case(db,actor,case_id)
            existing = db.execute('SELECT id,status FROM jobs WHERE digest=?',(digest,)).fetchone()
            if existing:
                return dict(existing)
            pending = db.execute("SELECT COUNT(*) FROM jobs WHERE status IN ('pending','running')").fetchone()[0]
            if pending >= 32:
                raise ValueError('The local queue is full (32 jobs). Wait for processing to finish.')
            job_id = 'job-'+uuid.uuid4().hex
            db.execute('INSERT INTO jobs(id,actor,case_id,kind,status,payload_json,digest,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?)',
                       (job_id,actor,case_id,kind,'pending',json.dumps(payload),digest,_now(),_now()))
        return {'id':job_id,'status':'pending'}

    def make_draft(self, actor, case_id):
        with self._connection(write=True) as db:
            case=self._case(db,actor,case_id)
            segments=db.execute('SELECT * FROM segments WHERE case_id=? ORDER BY position',(case_id,)).fetchall()
            documents=db.execute('SELECT * FROM documents WHERE case_id=? ORDER BY created_at',(case_id,)).fetchall()
            items=list(segments)+list(documents)
            if not items or any(not item['reviewed'] for item in items):
                raise ValueError('Review every transcript segment and document before preparing a draft.')
            sources=[{'id':s['id'],'title':'Reviewed transcript','text':s['text'],'start':s['start'],'end':s['end'],'case_id':case_id} for s in segments]
            sources.extend({'id':d['id'],'title':d['name'],'text':d['text'],'start':None,'end':None,'case_id':case_id} for d in documents)
            narrative='\n\n'.join((f"[{_stamp(s['start'])}] " if s['start'] is not None else f"[{s['title']}] ")+s['text'] for s in sources)
            revision=case['revision']+1
            draft={'text':narrative,'narrative':narrative,'sources':sources,'revision':revision,'based_on_revision':case['revision'],'created_at':_now(),'mode':'extractive','notice':'Reviewed source excerpts; not verified facts or an official record.'}
            db.execute("UPDATE cases SET revision=?,status='draft',draft_json=?,approved_revision=NULL,approved_by=NULL,updated_at=? WHERE id=?",(revision,json.dumps(draft,ensure_ascii=False),_now(),case_id))
            self._audit(db,case_id,actor,'draft_created',revision,{'content_hash':_hash(narrative),'source_ids':[s['id'] for s in sources],'based_on_revision':case['revision']})
        return self.get_case(actor,case_id)

    def approve(self, actor, case_id, expected_revision):
        with self._connection(write=True) as db:
            case=self._case(db,actor,case_id)
            if actor!='reviewer':
                raise PermissionError('Only the simulated reviewer may approve a draft.')
            self._revision(case,expected_revision)
            sources=list(db.execute('SELECT reviewed,reviewed_by FROM segments WHERE case_id=?',(case_id,)))+list(db.execute('SELECT reviewed,reviewed_by FROM documents WHERE case_id=?',(case_id,)))
            if not sources or any(not s['reviewed'] for s in sources):
                raise ValueError('Review every source before approval.')
            if any(s['reviewed_by']==actor for s in sources):
                raise PermissionError('The reviewer cannot approve their own source edits.')
            draft=json.loads(case['draft_json']) if case['draft_json'] else None
            if not draft or draft['revision']!=case['revision']:
                raise ValueError('Prepare a current draft before approval.')
            revision=case['revision']+1
            db.execute("UPDATE cases SET revision=?,status='approved',approved_revision=?,approved_by=?,updated_at=? WHERE id=?",(revision,draft['revision'],actor,_now(),case_id))
            self._audit(db,case_id,actor,'draft_approved',revision,{'draft_revision':draft['revision'],'content_hash':_hash(draft['text'])})
        return self.get_case(actor,case_id)

    def get_job(self,actor,job_id):
        self._actor(actor)
        with self._connection() as db:
            row = db.execute('SELECT * FROM jobs WHERE id=?',(job_id,)).fetchone()
            if row is None:
                raise KeyError('Job not found.')
            if row['case_id']:
                self._case(db,actor,row['case_id'])
            elif actor != row['actor'] and actor != 'reviewer':
                raise PermissionError('This simulated actor cannot access this job.')
            return {'job_id':row['id'],'status':row['status'],'kind':row['kind'],'result':json.loads(row['result_json']) if row['result_json'] else None,'error':row['error']}

    def job_update(self,job_id,status,result=None,error=None):
        with self._connection(write=True) as db:
            db.execute('UPDATE jobs SET status=?,result_json=?,error=?,updated_at=? WHERE id=?',
                       (status,json.dumps(result) if result is not None else None,error,_now(),job_id))
