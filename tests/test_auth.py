"""Real local credentials, session binding, forgery, expiry and worker revocation."""
import json
import time

import pytest
from fastapi.testclient import TestClient

from app import create_app
from lab.auth import AuthenticationError, LocalSessions, password_hash, token_hash


@pytest.fixture(scope="module")
def encoded():
    return password_hash("correct-local-test-passphrase")


@pytest.fixture
def authenticated(tmp_path, monkeypatch, encoded):
    credentials=tmp_path / "credentials.json"
    entries={actor:{"enabled":True,"password_hash":encoded} for actor in ("officer-a","officer-b","reviewer")}
    credentials.write_text(json.dumps(entries))
    monkeypatch.setenv("LAB_AUTH_MODE","local_sessions")
    monkeypatch.setenv("LAB_CREDENTIAL_FILE",str(credentials))
    app=create_app(tmp_path / "data")
    with TestClient(app) as client:
        yield client,app,credentials,entries


def sign_in(client, username="officer-a"):
    response=client.post("/api/auth/login",json={"username":username,"password":"correct-local-test-passphrase"})
    assert response.status_code==200
    return response.json()["csrf"]


def test_no_actor_selector_bypass_and_no_password_disclosure(authenticated):
    client,app,_,_=authenticated
    assert client.get("/api/cases?actor=reviewer").status_code==401
    csrf=sign_in(client)
    assert client.get("/api/cases").status_code==200
    assert client.get("/api/cases?actor=reviewer").status_code==403
    assert client.get("/api/cases/demo-case-b").status_code==403
    assert client.post("/api/models/train",headers={"X-Lab-CSRF":csrf}).status_code==403
    actor=client.get("/api/actors").json()["actors"]
    assert len(actor)==1 and actor[0]["id"]=="officer-a" and not actor[0]["simulated_identity"]
    assert not client.get("/api/cases/demo-case-a").json()["simulated_identity"]
    with app.state.auth.connection() as db:
        session=dict(db.execute("SELECT * FROM sessions").fetchone())
    assert csrf not in str(session) and client.cookies.get("lab_session") not in str(session)


def test_mutations_require_csrf_and_logout_revokes(authenticated):
    client,_,_,_=authenticated
    csrf=sign_in(client)
    assert client.post("/api/cases",json={"title":"Denied mutation"}).status_code==401
    assert client.post("/api/cases",json={"title":"Accepted mutation"},headers={"X-Lab-CSRF":csrf}).status_code==200
    old=client.cookies.get("lab_session")
    assert client.post("/api/auth/logout",headers={"X-Lab-CSRF":csrf}).status_code==200
    client.cookies.set("lab_session",old)
    assert client.get("/api/cases").status_code==401


def test_cookie_flags_and_expiry_survive_process_reopen(authenticated):
    client,app,path,_=authenticated
    response=client.post("/api/auth/login",json={"username":"reviewer","password":"correct-local-test-passphrase"})
    assert "HttpOnly" in response.headers["set-cookie"] and "SameSite=strict" in response.headers["set-cookie"]
    token=client.cookies.get("lab_session")
    reopened=LocalSessions(app.state.store.root,path)
    assert reopened.authenticate(token)["id"]=="reviewer"
    with reopened.connection() as db:
        db.execute("UPDATE sessions SET expires=?",(time.time()-1,))
    assert client.get("/api/cases").status_code==401


def test_account_revocation_blocks_session_and_worker(authenticated):
    client,app,path,entries=authenticated
    sign_in(client)
    entries["officer-a"]["enabled"]=False
    path.write_text(json.dumps(entries))
    assert client.get("/api/cases").status_code==401
    with pytest.raises(PermissionError):
        app.state.auth.authorize_job("officer-a","document")
    with pytest.raises(PermissionError):
        app.state.store.get_case("officer-a","demo-case-a")


def test_invalid_credentials_are_bounded(authenticated):
    client,app,_,_=authenticated
    for _ in range(5):
        assert client.post("/api/auth/login",json={"username":"officer-a","password":"incorrect"}).status_code==401
    assert "temporarily" in client.post("/api/auth/login",json={"username":"officer-a","password":"correct-local-test-passphrase"}).json()["detail"]
    with app.state.auth.connection() as db:
        assert db.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]==0


def test_missing_auth_configuration_never_falls_back(tmp_path,monkeypatch):
    monkeypatch.setenv("LAB_AUTH_MODE","local_sessions")
    monkeypatch.setenv("LAB_CREDENTIAL_FILE",str(tmp_path/"missing.json"))
    with pytest.raises(FileNotFoundError):
        create_app(tmp_path / "data")
