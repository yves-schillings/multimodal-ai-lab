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
    """Hash one fixed test-only passphrase once for the module.

    Returns:
        Password hash consumed by isolated test credentials; no real account is used.
    """
    return password_hash("engineering-test-passphrase")


@pytest.fixture
def secured(tmp_path, monkeypatch, encoded):
    """Prepare isolated enabled accounts and disable optional agent configuration.

    Args:
        tmp_path: pytest Path for isolated temporary case/session files; no production data.
        monkeypatch: pytest fixture that restores patched clients, methods and environment values.
        encoded: Module-scoped hash of a fixed test-only passphrase, not a real credential.

    Returns:
        Tuple (temporary root Path, credentials Path, mutable account mapping).

    Notes:
        Writes only temporary credentials. monkeypatch restores environment values
        after the test; credentials are never delivered or published.
    """
    entries = {name: {"enabled": True, "password_hash": encoded}
               for name in ("officer-a", "officer-b", "reviewer")}
    credentials = tmp_path / "credentials.json"
    credentials.write_text(json.dumps(entries))
    monkeypatch.setenv("LAB_AUTH_MODE", "local_sessions")
    monkeypatch.setenv("LAB_CREDENTIAL_FILE", str(credentials))
    monkeypatch.delenv("LAB_ENGINEERING_AGENTIC", raising=False)
    return tmp_path, credentials, entries


def login(client, name="officer-a"):
    """Authenticate a fixed synthetic account through the in-process test client.

    Args:
        client: FastAPI TestClient for the isolated application.
        name: Synthetic username; defaults to officer-a.

    Returns:
        Header dictionary containing the returned X-Lab-CSRF token.

    Raises:
        AssertionError: The test login did not return HTTP 200.

    Notes:
        Establishes only a test-client cookie/session; no external service is called.
    """
    response = client.post("/api/auth/login", json={"username": name, "password": "engineering-test-passphrase"})
    assert response.status_code == 200
    return {"X-Lab-CSRF": response.json()["csrf"]}


def test_study_routes_require_session_csrf_and_case_access(secured):
    """Verify session, CSRF, actor binding and case ownership across study routes.

    Args:
        secured: Fixture containing isolated store root, temporary credentials and mutable test accounts.

    Returns:
        None; pytest assertions or expected exceptions determine success.

    Raises:
        AssertionError: The behavior or access-control invariant regresses.

    Notes:
        Uses isolated fixtures and scripted clients; passing does not measure
        real model quality, service capacity or production authorization.
    """
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
    """Withhold an agent answer when its account is disabled during work.

    Args:
        secured: Fixture containing isolated store root, temporary credentials and mutable test accounts.
        monkeypatch: pytest fixture that restores patched clients, methods and environment values.

    Returns:
        None; pytest assertions or expected exceptions determine success.

    Raises:
        AssertionError: The behavior or access-control invariant regresses.

    Notes:
        Uses isolated fixtures and scripted clients; passing does not measure
        real model quality, service capacity or production authorization.
    """
    root, credentials, entries = secured

    class RevokingAgent:
        """Fake agent that revokes the current account between route access checks.
        """
        def answer(self, store, actor, case_id, question, reauthorize):
            """Disable the current synthetic account just before returning a forbidden answer.

            Args:
                store: Unused isolated case store.
                actor: Test account to disable in the enclosing credential mapping.
                case_id: Unused fixed case identifier.
                question: Unused original question.
                reauthorize: Callback checked before account revocation.

            Returns:
                Sentinel answer that the route's final access check must withhold.

            Notes:
                Writes the enclosing temporary credential file to simulate revocation.
            """
            reauthorize()
            entries[actor]["enabled"] = False
            credentials.write_text(json.dumps(entries))
            return {"answer": "This answer must never be released."}

    # Configuration callback: rag is ignored; install the scripted revoking agent only.
    monkeypatch.setattr(engineering_api, "_agentic_configuration", lambda rag: RevokingAgent())
    with TestClient(create_app(root / "data")) as client:
        headers = login(client)
        response = client.post("/api/cases/demo-case-a/agent-question?actor=officer-a",
                               json={"question": "Where was the bicycle?"}, headers=headers)
        assert response.status_code == 403
        assert "never be released" not in response.text


def test_caller_cannot_choose_model_identity_or_endpoint(secured):
    """Reject extra client-controlled agent settings and non-reviewer explanation calls.

    Args:
        secured: Fixture containing isolated store root, temporary credentials and mutable test accounts.

    Returns:
        None; pytest assertions or expected exceptions determine success.

    Raises:
        AssertionError: The behavior or access-control invariant regresses.

    Notes:
        Uses isolated fixtures and scripted clients; passing does not measure
        real model quality, service capacity or production authorization.
    """
    root, _, _ = secured
    with TestClient(create_app(root / "data")) as client:
        headers = login(client)
        response = client.post("/api/cases/demo-case-a/agent-question?actor=officer-a",
                               json={"question": "Where?", "endpoint": "https://untrusted.invalid", "actor": "reviewer"},
                               headers=headers)
        assert response.status_code == 422
        assert client.post("/api/engineering/explain?actor=officer-a", json={"invoice": None}, headers=headers).status_code == 403


def test_retained_vision_does_not_expose_unlisted_raw_fields(secured, monkeypatch):
    """Return only the listed vision fields and label the result as retained evidence.

    Args:
        secured: Fixture containing isolated store root, temporary credentials and mutable test accounts.
        monkeypatch: pytest fixture that restores patched clients, methods and environment values.

    Returns:
        None; pytest assertions or expected exceptions determine success.

    Raises:
        AssertionError: The behavior or access-control invariant regresses.

    Notes:
        Uses isolated fixtures and scripted clients; passing does not measure
        real model quality, service capacity or production authorization.
    """
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
    """Return HTTP 503 instead of silently substituting an unconfigured agent.

    Args:
        secured: Fixture containing isolated store root, temporary credentials and mutable test accounts.

    Returns:
        None; pytest assertions or expected exceptions determine success.

    Raises:
        AssertionError: The behavior or access-control invariant regresses.

    Notes:
        Uses isolated fixtures and scripted clients; passing does not measure
        real model quality, service capacity or production authorization.
    """
    root, _, _ = secured
    with TestClient(create_app(root / "data")) as client:
        headers = login(client)
        response = client.post("/api/cases/demo-case-a/agent-question?actor=officer-a",
                               json={"question": "Where?"}, headers=headers)
        assert response.status_code == 503
        assert "not configured" in response.json()["detail"]


def test_existing_lexical_cpe_does_not_implicitly_approve_model_fitting(secured, monkeypatch):
    """Prevent a lexical controlled-project profile from approving synthetic fitting.

    Args:
        secured: Fixture containing isolated store root, temporary credentials and mutable test accounts.
        monkeypatch: pytest fixture that restores patched clients, methods and environment values.

    Returns:
        None; pytest assertions or expected exceptions determine success.

    Raises:
        AssertionError: The behavior or access-control invariant regresses.

    Notes:
        Uses isolated fixtures and scripted clients; passing does not measure
        real model quality, service capacity or production authorization.
    """
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
