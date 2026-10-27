"""The optional study view retains the base service's actual access controls."""
import json
import time

import pytest
from fastapi.testclient import TestClient

from app import create_app
from lab import engineering_api
from lab.auth import password_hash


@pytest.fixture(scope="module")
def encoded():
    return password_hash("engineering-test-passphrase")


@pytest.fixture
def secured(tmp_path, monkeypatch, encoded):
    entries = {name: {"enabled": True, "password_hash": encoded}
               for name in ("officer-a", "officer-b", "reviewer")}
    credentials = tmp_path / "credentials.json"
    credentials.write_text(json.dumps(entries))
    monkeypatch.setenv("LAB_AUTH_MODE", "local_sessions")
    monkeypatch.setenv("LAB_CREDENTIAL_FILE", str(credentials))
    monkeypatch.delenv("LAB_ENGINEERING_AGENTIC", raising=False)
    return tmp_path, credentials, entries


def login(client, name="officer-a"):
    response = client.post("/api/auth/login", json={"username": name, "password": "engineering-test-passphrase"})
    assert response.status_code == 200
    return {"X-Lab-CSRF": response.json()["csrf"]}


def test_study_routes_require_session_csrf_and_case_access(secured):
    root, _, _ = secured
    with TestClient(create_app(root / "data")) as client:
        assert client.get("/engineering").status_code == 200
        assert client.get("/api/engineering/status?actor=reviewer").status_code == 401
        headers = login(client)
        assert client.get("/api/engineering/status?actor=reviewer").status_code == 403
        assert client.post("/api/engineering/example?actor=officer-a", json={}).status_code == 401
        response = client.post("/api/engineering/example?actor=officer-a", json={}, headers=headers)
        assert response.status_code == 200
        case = response.json()["case"]
        assert len(case["documents"]) == 2
        assert all(item["reviewed"] and item["engine"] == "synthetic_engineering_fixture" for item in case["documents"])
        login(client, "officer-b")
        assert client.get(f"/api/cases/{case['id']}?actor=officer-b").status_code == 403


def test_agent_rechecks_real_session_before_returning_answer(secured, monkeypatch):
    root, credentials, entries = secured

    class RevokingAgent:
        def answer(self, store, actor, case_id, question, reauthorize):
            reauthorize()
            entries[actor]["enabled"] = False
            credentials.write_text(json.dumps(entries))
            return {"answer": "This answer must never be released."}

    monkeypatch.setattr(engineering_api, "_agentic_configuration", lambda rag: RevokingAgent())
    with TestClient(create_app(root / "data")) as client:
        headers = login(client)
        response = client.post("/api/cases/demo-case-a/agent-question?actor=officer-a",
                               json={"question": "Where was the bicycle?"}, headers=headers)
        assert response.status_code == 403
        assert "never be released" not in response.text


def test_caller_cannot_choose_model_identity_or_endpoint(secured):
    root, _, _ = secured
    with TestClient(create_app(root / "data")) as client:
        headers = login(client)
        response = client.post("/api/cases/demo-case-a/agent-question?actor=officer-a",
                               json={"question": "Where?", "endpoint": "https://untrusted.invalid", "actor": "reviewer"},
                               headers=headers)
        assert response.status_code == 422
        assert client.post("/api/engineering/explain?actor=officer-a", json={"invoice": None}, headers=headers).status_code == 403


def test_retained_vision_does_not_expose_unlisted_raw_fields(secured, monkeypatch):
    root, _, _ = secured
    report = root / "report.json"
    report.write_text(json.dumps({"schema_version": 1, "scope": "synthetic benchmark",
                                  "benchmarks": [], "private_file": "do-not-expose", "fixture_pixels": [1, 2]}))
    monkeypatch.setenv("LAB_ENGINEERING_VISION_REPORT", str(report))
    with TestClient(create_app(root / "data")) as client:
        login(client)
        response = client.get("/api/engineering/vision?actor=officer-a")
        assert response.status_code == 200
        data = response.json()
        assert data["view"] == "retained_measured_run" and data["live_gpu_inference"] is False
        assert len(data["report_sha256"]) == 64
        assert "do-not-expose" not in response.text and "fixture_pixels" not in response.text


def test_unconfigured_agent_is_explicit_not_a_silent_lexical_answer(secured):
    root, _, _ = secured
    with TestClient(create_app(root / "data")) as client:
        headers = login(client)
        response = client.post("/api/cases/demo-case-a/agent-question?actor=officer-a",
                               json={"question": "Where?"}, headers=headers)
        assert response.status_code == 503
        assert "not configured" in response.json()["detail"]


def test_existing_lexical_cpe_does_not_implicitly_approve_model_fitting(secured, monkeypatch):
    root, _, _ = secured
    control = root / "cpe.json"
    control.write_text(json.dumps({"schema": 1, "phase": "active", "approval": "synthetic test approval",
                                   "expires": time.time() + 60, "models": [], "profile_digest": "a" * 64,
                                   "evidence_sha256": "b" * 64}))
    monkeypatch.setenv("LAB_CPE_STATE_FILE", str(control))
    monkeypatch.setenv("LAB_RETRIEVAL_BACKEND", "lexical")
    with TestClient(create_app(root / "data")) as client:
        headers = login(client, "reviewer")
        status = client.get("/api/engineering/status?actor=reviewer").json()
        assert status["can_run_explainability"] is False
        response = client.post("/api/engineering/explain?actor=reviewer", json={"invoice": None}, headers=headers)
        assert response.status_code == 403
        assert "does not approve" in response.json()["detail"]
