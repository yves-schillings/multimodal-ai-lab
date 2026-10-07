"""Fail-closed network mode: loopback by default, container mode only when explicitly set."""
import pytest
from fastapi.testclient import TestClient

from app import create_app, network_settings


def test_default_mode_is_loopback(monkeypatch):
    monkeypatch.delenv("LAB_NETWORK_MODE", raising=False)
    monkeypatch.delenv("LAB_ALLOWED_HOSTS", raising=False)
    mode, allowed = network_settings()
    assert mode == "loopback"
    assert allowed == {"127.0.0.1", "localhost", "::1"}


def test_unknown_mode_is_refused(monkeypatch):
    monkeypatch.setenv("LAB_NETWORK_MODE", "public")
    with pytest.raises(RuntimeError):
        network_settings()


def test_container_mode_still_checks_host_header(monkeypatch, tmp_path):
    monkeypatch.setenv("LAB_NETWORK_MODE", "container")
    monkeypatch.setenv("LAB_ALLOWED_HOSTS", "lab.internal, Lab-Route.Example")
    with TestClient(create_app(tmp_path)) as client:
        status = client.get("/api/status").json()
        assert status["network_mode"] == "container"
        assert status["simulated_identity"] is True
        assert "lab.internal" in status["allowed_hosts"]
        assert client.get("/api/status", headers={"Host": "lab.internal"}).status_code == 200
        assert client.get("/api/status", headers={"Host": "lab-route.example"}).status_code == 200
        assert client.get("/api/status", headers={"Host": "evil.example"}).status_code == 403
        assert client.get("/health/ready").json()["network_mode"] == "container"


def test_loopback_mode_keeps_host_guard(monkeypatch, tmp_path):
    monkeypatch.setenv("LAB_NETWORK_MODE", "loopback")
    monkeypatch.delenv("LAB_ALLOWED_HOSTS", raising=False)
    with TestClient(create_app(tmp_path)) as client:
        assert client.get("/api/status", headers={"Host": "lab.internal"}).status_code == 403
        assert client.get("/api/status").json()["network_mode"] == "loopback"
