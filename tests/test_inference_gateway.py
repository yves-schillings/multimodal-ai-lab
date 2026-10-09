import json
import pytest
from fastapi.testclient import TestClient
from lab.inference_gateway import create_gateway


@pytest.fixture
def gateway(tmp_path):
    path=tmp_path/'tokens.json';path.write_text(json.dumps({'a':'synthetic-token-a','b':'synthetic-token-b'}))
    with TestClient(create_gateway(path)) as client:
        yield client


def test_missing_wrong_token_and_forbidden_operation(gateway):
    assert gateway.get('/api/tags').status_code==401
    assert gateway.get('/api/tags',headers={'Authorization':'Bearer wrong'}).status_code==401
    gateway.headers['Authorization']='Bearer synthetic-token-a'
    assert gateway.post('/api/pull',json={'model':'qwen3:4b'}).status_code==403
    assert gateway.post('/api/generate',json={'model':'cloud-model'}).status_code==403
    assert gateway.post('/api/embed',json={'model':'qwen3:4b'}).status_code==403


@pytest.mark.parametrize('payload',[[],{'model':'bge-m3:latest','input':['x'*2001]},
                                   {'model':'bge-m3:latest','input':['x']*129}])
def test_malformed_or_unbounded_never_contacts_model(gateway,payload):
    gateway.headers['Authorization']='Bearer synthetic-token-a'
    assert gateway.post('/api/embed',json=payload).status_code==400


def test_rejected_requests_consume_bounded_project_budget(gateway):
    gateway.headers['Authorization']='Bearer synthetic-token-a'
    for _ in range(32):assert gateway.post('/api/pull').status_code==403
    # Allowed operations are budgeted even if malformed.
    for _ in range(32):assert gateway.post('/api/embed',json=[]).status_code==400
    assert gateway.post('/api/embed',json=[]).status_code==429
    assert gateway.post('/api/embed',json=[],headers={'Authorization':'Bearer synthetic-token-b'}).status_code==400
