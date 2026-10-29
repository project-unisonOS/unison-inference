import pathlib
import sys

import pytest
from fastapi.testclient import TestClient

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

import server  # noqa: E402


@pytest.fixture(autouse=True)
def reset_settings(monkeypatch):
    # Ensure defaults are loaded fresh for each test
    server.SETTINGS = server.InferenceServiceSettings.from_env()
    yield


def test_unauthenticated_multimodal_request_cannot_reach_provider(monkeypatch):
    captured = {}

    def fake_provider(provider, model, messages, **kwargs):
        captured.update({"provider": provider, "model": model, "messages": messages})
        return {"content": "ok", "messages": messages}

    monkeypatch.setattr(server, "_call_provider", fake_provider)
    monkeypatch.setattr(server, "_provider_ready_status", lambda p, m=None: (True, "ok"))

    client = TestClient(server.app)
    resp = client.post(
        "/inference/request",
        json={
            "intent": "companion.turn",
            "messages": [{"role": "user", "content": "see"}],
            "attachments": [{"type": "image", "data": "fake"}],
        },
    )

    assert resp.status_code == 401
    assert captured == {}


def test_unauthenticated_fallback_request_is_rejected(monkeypatch):
    calls = {"ready": []}

    def fake_ready(provider, model=None):
        calls["ready"].append((provider, model))
        if provider == "ollama":
            return False, "unreachable"
        return True, "ok"

    def fake_provider(provider, model, messages, **kwargs):
        return {"content": "ok", "messages": messages}

    monkeypatch.setattr(server, "_provider_ready_status", fake_ready)
    monkeypatch.setattr(server, "_call_provider", fake_provider)

    client = TestClient(server.app)
    resp = client.post(
        "/inference/request",
        json={
            "intent": "companion.turn",
            "messages": [{"role": "user", "content": "hi"}],
            "allow_fallback": True,
            "fallback_provider": "openai",
            "fallback_model": "gpt-4o-mini",
            "local_alternative_checked": True,
            "trust_decision": {
                "outcome": "minimize",
                "disclosed_fields": ["intent", "messages"],
            },
        },
    )

    assert resp.status_code == 401
    assert calls["ready"] == []
