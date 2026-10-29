import pytest
import server
from authenticated_fixture import authenticated_inference
from test_governed_models_se8_se10 import registry_with_local


def payload():
    return {"intent": "synthetic-summary", "prompt": "Synthetic fixture only.",
            "task_requirement": {"task": "interpretation", "local_only": True},
            "hardware": {"architecture": "x86_64", "ram_mb": 4096, "storage_mb": 1000}}


@pytest.fixture
def controlled_model(monkeypatch):
    _, registry = registry_with_local()
    monkeypatch.setattr(server, "_GOVERNED_REGISTRY", registry)
    monkeypatch.setattr(server, "_provider_ready_status", lambda *a: (True, "synthetic"))
    monkeypatch.setattr(server.app.state, "model_hardware", {"architecture": "x86_64", "ram_mb": 4096, "storage_mb": 1000}, raising=False)
    calls = []
    def invoke(provider, model, messages, **kwargs):
        calls.append((provider, model, messages))
        return {"content": "Synthetic result", "messages": []}
    monkeypatch.setattr(server, "_call_provider", invoke)
    return calls


@pytest.mark.parametrize("case", ["missing", "scope", "expired", "revoked", "audience", "forged"])
def test_inference_denies_without_person_authority(authenticated_inference, controlled_model, case):
    client, headers, revoked, _ = authenticated_inference
    request = payload()
    authorization = headers()
    if case == "missing": authorization = {}
    if case == "scope": authorization = headers(scopes=[])
    if case == "expired": authorization = headers(exp=1)
    if case == "revoked": revoked.add("session-alice")
    if case == "audience": authorization = headers(aud=["orchestrator"])
    if case == "forged": request["person_id"] = "bob"
    response = client.post("/inference/request", json=request, headers=authorization)
    assert response.status_code in (401, 403)
    assert controlled_model == []


def test_legacy_request_has_no_provider_side_effect(authenticated_inference, controlled_model):
    client, headers, _, _ = authenticated_inference
    response = client.post("/inference/request", json={"intent": "fixture", "prompt": "synthetic"}, headers=headers())
    assert response.status_code == 400
    assert controlled_model == []


def test_local_unavailability_cannot_escape_to_caller_cloud_fallback(authenticated_inference, controlled_model, monkeypatch):
    client, headers, _, _ = authenticated_inference
    monkeypatch.setattr(server, "_provider_ready_status", lambda p, *a: (p != "ollama", "synthetic"))
    body = {**payload(), "allow_fallback": True, "fallback_provider": "openai", "fallback_model": "unreviewed",
            "local_alternative_checked": True, "trust_decision": {"outcome": "allow", "disclosed_fields": ["prompt"]},
            "hardware": {"architecture": "x86_64", "ram_mb": 4096, "storage_mb": 1000}}
    response = client.post("/inference/request", json=body, headers=headers())
    assert response.status_code == 503
    assert controlled_model == []


def test_authenticated_local_read_only_request(authenticated_inference, controlled_model):
    client, headers, _, _ = authenticated_inference
    response = client.post("/inference/request", json=payload(), headers=headers())
    assert response.status_code == 200
    assert response.json()["provider"] == "ollama"
    assert len(controlled_model) == 1


@pytest.mark.parametrize("extra", [
    {"provider": "openai", "trust_decision": {"outcome": "allow"}},
    {"tools": [{"name": "untrusted-tool"}]},
    {"attachments": [{"type": "image", "data": "synthetic"}]},
    {"max_tokens": 1025}, {"max_tokens": -1},
])
def test_local_scope_cannot_expand_capabilities(authenticated_inference, controlled_model, extra):
    client, headers, _, _ = authenticated_inference
    response = client.post("/inference/request", json={**payload(), **extra}, headers=headers())
    assert response.status_code in (400, 403)
    assert controlled_model == []


def test_request_hardware_cannot_replace_trusted_inventory(authenticated_inference, controlled_model, monkeypatch):
    client, headers, _, _ = authenticated_inference
    monkeypatch.setattr(server.app.state, "model_hardware", {})
    response = client.post("/inference/request", json=payload(), headers=headers())
    assert response.status_code == 503
    assert controlled_model == []


def test_registry_is_required_even_when_optional_startup_flag_is_false(authenticated_inference, controlled_model, monkeypatch):
    client, headers, _, _ = authenticated_inference
    monkeypatch.setattr(server, "_GOVERNED_REGISTRY", None)
    response = client.post("/inference/request", json=payload(), headers=headers())
    assert response.status_code == 503
    assert controlled_model == []
