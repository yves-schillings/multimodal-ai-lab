import json
import time
import pytest
from fastapi.testclient import TestClient
from app import create_app


@pytest.fixture
def client(tmp_path):
    application=create_app(tmp_path)
    with TestClient(application) as client:
        yield client


def wait_job(client,job_id,actor='officer-a'):
    for _ in range(200):
        response=client.get(f'/api/jobs/{job_id}',params={'actor':actor})
        assert response.status_code==200,response.text
        job=response.json()
        if job['status'] in {'completed','failed'}:
            return job
        time.sleep(.05)
    raise AssertionError('Job did not finish.')


def test_case_access_and_identity(client):
    assert client.get('/api/status').json()['simulated_identity']
    assert client.get('/api/cases/demo-case-b?actor=officer-a').status_code==403
    assert client.get('/api/cases?actor=unknown').status_code==403
    assert client.get('/api/cases').status_code==422
    cases=client.get('/api/cases?actor=officer-a').json()['cases']
    assert {c['id'] for c in cases}=={'demo-case-a'}


def test_host_and_origin_guard(client):
    assert client.get('/api/status',headers={'Host':'evil.example'}).status_code==403
    assert client.post('/api/cases?actor=officer-a',json={'title':'example'},headers={'Origin':'https://evil.example'}).status_code==403
    assert client.post('/api/cases?actor=officer-a',json={'title':'example'},headers={'Origin':'http://localhost:1234'}).status_code==403


def test_review_conflict_and_separate_approval(client):
    case=client.get('/api/cases/demo-case-a?actor=officer-a').json()
    initial_revision=case['revision']
    for segment in case['segments']:
        result=client.patch(f"/api/cases/demo-case-a/segments/{segment['id']}?actor=officer-a",json={'text':segment['text'],'expected_revision':case['revision']})
        assert result.status_code==200,result.text
        case=result.json()
    assert client.patch(f"/api/cases/demo-case-a/segments/{case['segments'][0]['id']}?actor=officer-a",json={'text':'New text','expected_revision':initial_revision}).status_code==409
    case=client.post('/api/cases/demo-case-a/draft?actor=officer-a',json={}).json()
    assert client.post('/api/cases/demo-case-a/approve?actor=officer-a',json={'expected_revision':case['revision']}).status_code==403
    approved=client.post('/api/cases/demo-case-a/approve?actor=reviewer',json={'expected_revision':case['revision']})
    assert approved.status_code==200,approved.text
    assert approved.json()['status']=='approved'


def test_invalid_upload_and_authorization_before_job(client):
    assert client.post('/api/cases/demo-case-b/upload?actor=officer-a',files={'file':('a.txt',b'x')}).status_code==403
    assert client.post('/api/cases/demo-case-a/upload?actor=officer-a',files={'file':('a.exe',b'x')}).status_code==422
    assert client.post('/api/cases/demo-case-a/upload?actor=officer-a',files={'file':('a.txt',b'')}).status_code==422


def test_document_job_review_retrieval_and_idempotency(client):
    def upload():
        return client.post('/api/cases/demo-case-a/upload?actor=officer-a',files={'file':('test.txt',b'Evidence inventory records a purple camera exhibit.')},data={'kind':'document','language':'en'})
    response=upload()
    assert response.status_code==200,response.text
    job_id=response.json()['job_id']
    assert client.get(f'/api/jobs/{job_id}?actor=officer-b').status_code==403
    job=wait_job(client,job_id)
    assert job['status']=='completed',job
    assert upload().json()['job_id']==job_id
    case=client.get('/api/cases/demo-case-a?actor=officer-a').json()
    assert len(case['documents'])==1
    document=case['documents'][0]
    answer=client.post('/api/cases/demo-case-a/question?actor=officer-a',json={'question':'purple camera'}).json()
    assert not any(s.get('document_id') for s in answer['sources'])
    response=client.post(f"/api/cases/demo-case-a/documents/{document['id']}/review?actor=officer-a",json={'expected_revision':case['revision']})
    assert response.status_code==200,response.text
    answer=client.post('/api/cases/demo-case-a/question?actor=officer-a',json={'question':'purple camera'}).json()
    assert any(s.get('document_id')==document['id'] for s in answer['sources'])
    updated=response.json()
    assert updated['documents'][0]['original_text']==document['text']


def test_bounded_workflow_never_approves(client):
    result=client.post('/api/cases/demo-case-a/workflow?actor=officer-a',json={'question':'What is review?'}).json()
    assert result['mode']=='bounded_deterministic_workflow'
    assert len(result['steps'])<=result['max_steps']
    assert result['case']['status']!='approved'
    assert result['steps'][2]['status']=='blocked'


def test_recovered_ingestion_is_idempotent(client):
    store=client.app.state.store
    extraction={'text':'Synthetic exhibit inventory.','engine':'utf8'}
    first=store.add_document('officer-a','demo-case-a','example.txt',extraction,{},'recovery-test')
    second=store.add_document('officer-a','demo-case-a','example.txt',extraction,{},'recovery-test')
    assert second==first
    assert len(store.get_case('officer-a','demo-case-a')['documents'])==1


def test_model_gate_reviewer_only_and_rollback(client):
    registry=client.app.state.registry
    first=registry.train()
    assert first['gate_passed'],first
    assert first['accuracy']>=.85 and first['test_samples']==8
    assert client.post('/api/models/promote?actor=officer-a',json={}).status_code==403
    assert client.post('/api/models/promote?actor=reviewer',json={'version':first['version']}).status_code==200
    assert registry.classify('Dear colleagues please reply to this email')['label']=='correspondence'
    second=registry.train()
    registry.promote('reviewer',second['version'])
    assert client.post('/api/models/rollback?actor=reviewer',json={}).json()['active_version']==first['version']
    second['gate_passed']=False
    with client.app.state.store._connection(write=True) as db:
        db.execute('UPDATE model_versions SET metadata_json=? WHERE version=?',(json.dumps(second),second['version']))
    assert client.post('/api/models/promote?actor=reviewer',json={'version':second['version']}).status_code==422
