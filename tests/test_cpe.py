import json
import time
import pytest
from fastapi.testclient import TestClient
from app import create_app
from lab.auth import password_hash
from lab.cpe import CpeGate,CpeInactive


def control(path,phase='active',**overrides):
    data={'schema':1,'phase':phase,'approval':'synthetic-owner-approval','expires':time.time()+3600,
          'profile_digest':'a'*64,'evidence_sha256':'b'*64,'models':[]}
    path.write_text(json.dumps({**data,**overrides}))


def test_gate_fails_closed_and_checks_each_call(tmp_path):
    path=tmp_path/'control.json';gate=CpeGate(path)
    with pytest.raises(CpeInactive):gate.authorize('officer-a')
    control(path,'provisioned')
    with pytest.raises(CpeInactive):gate.authorize('reviewer')
    control(path,'verifying');gate.authorize('reviewer')
    with pytest.raises(CpeInactive):gate.authorize('officer-a')
    control(path);gate.authorize('officer-a')
    with pytest.raises(PermissionError):gate.authorize('reviewer','/api/models/train')
    control(path,'revoked')
    with pytest.raises(CpeInactive):gate.authorize('officer-a')


@pytest.mark.parametrize('changes',[{'expires':0},{'expires':float('nan')},{'evidence_sha256':None},
                                   {'profile_digest':'z'*64},{'models':['remote']},{'approval':''}])
def test_invalid_or_expired_control_denies(tmp_path,changes):
    path=tmp_path/'control.json';control(path,**changes)
    with pytest.raises(CpeInactive):CpeGate(path).authorize('reviewer')


@pytest.mark.parametrize('value',[None,[],"unavailable"])
def test_nonobject_operator_control_fails_closed(tmp_path,value):
    path=tmp_path/'control.json';path.write_text(json.dumps(value))
    assert CpeGate(path).state()['phase']=='unavailable'
    with pytest.raises(CpeInactive):CpeGate(path).authorize('reviewer')


def test_http_gate_revokes_existing_session_and_case_writes(tmp_path,monkeypatch):
    path=tmp_path/'control.json';control(path,'provisioned')
    credentials=tmp_path/'accounts.json';password='synthetic-test-password-123'
    credentials.write_text(json.dumps({a:{'enabled':True,'password_hash':password_hash(password)}
                                      for a in ['officer-a','reviewer']}))
    monkeypatch.setenv('LAB_AUTH_MODE','local_sessions');monkeypatch.setenv('LAB_CREDENTIAL_FILE',str(credentials))
    monkeypatch.setenv('LAB_CPE_STATE_FILE',str(path))
    with TestClient(create_app(tmp_path/'data')) as client:
        login=client.post('/api/auth/login',json={'username':'officer-a','password':password})
        assert login.status_code==200
        client.headers['X-Lab-CSRF']=login.json()['csrf']
        assert client.post('/api/cases',json={'title':'Synthetic gated case'}).status_code==503
        control(path)
        response=client.post('/api/cases',json={'title':'Synthetic gated case'});assert response.status_code==200
        case_id=response.json()['id']
        assert client.post('/api/cases/'+case_id+'/upload',
                           files={'file':('fictional.wav',b'no-speech-in-this-profile','audio/wav')},
                           data={'kind':'audio'}).status_code==403
        assert not (tmp_path/'data'/'uploads').exists()
        control(path,'revoked')
        assert client.get('/api/cases/'+case_id).status_code==503
        assert client.post('/api/cases',json={'title':'Must stay inactive'}).status_code==503
        assert client.get('/api/status').json()['cpe']['active'] is False


def test_cpe_cannot_start_with_simulated_identity(tmp_path,monkeypatch):
    monkeypatch.setenv('LAB_AUTH_MODE','simulated');monkeypatch.setenv('LAB_CPE_STATE_FILE',str(tmp_path/'missing'))
    with pytest.raises(RuntimeError,match='credential-bound'):create_app(tmp_path/'data')


def test_rag_requires_exact_operator_models_and_matching_runtime(tmp_path,monkeypatch):
    from lab.inference_gateway import MODELS
    path=tmp_path/'control.json'
    models=[{'name':name,'digest':digest} for name,digest in MODELS.items()]
    control(path,models=models)
    with pytest.raises(CpeInactive):CpeGate(path).authorize('reviewer')
    for k,v in {'LAB_RETRIEVAL_BACKEND':'pgvector','LAB_INFERENCE_URL':'http://lab-inference-gateway.multimodal-ai-lab.svc:8771',
                'LAB_EMBEDDING_MODEL':models[0]['name'],'LAB_EMBEDDING_DIGEST':models[0]['digest'],
                'LAB_GENERATION_MODEL':models[1]['name'],'LAB_GENERATION_DIGEST':models[1]['digest']}.items():
        monkeypatch.setenv(k,v)
    CpeGate(path).authorize('reviewer')
    control(path,models=[])
    with pytest.raises(CpeInactive):CpeGate(path).authorize('reviewer')
    control(path,models=models);monkeypatch.setenv('LAB_GENERATION_DIGEST','f'*64)
    with pytest.raises(CpeInactive):CpeGate(path).authorize('reviewer')
