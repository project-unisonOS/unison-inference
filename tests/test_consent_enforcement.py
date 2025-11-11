import pytest
from fastapi.testclient import TestClient
from fastapi import Request
import httpx

from unison_common.consent import ConsentScopes, clear_consent_cache
import os, sys


def make_consent_app():
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse

    app = FastAPI()

    @app.post("/introspect")
    async def introspect(request: Request):
        body = await request.json()
        token = body.get("token")
        if token == "valid-read":
            return JSONResponse({"active": True, "sub": "user1", "scopes": [ConsentScopes.REPLAY_READ]})
        if token == "admin":
            return JSONResponse({"active": True, "sub": "admin", "scopes": [ConsentScopes.ADMIN_ALL]})
        if token == "inactive":
            return JSONResponse({"active": False})
        return JSONResponse({"active": True, "scopes": []})

    return app


def test_inference_consent_enforced(monkeypatch):
    monkeypatch.setenv("UNISON_REQUIRE_CONSENT", "true")
    clear_consent_cache()
    consent_app = make_consent_app()
    consent_transport = httpx.ASGITransport(app=consent_app)

    orig_async_client = httpx.AsyncClient

    def _patched_async_client(*args, **kwargs):
        kwargs.setdefault("transport", consent_transport)
        return orig_async_client(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", _patched_async_client)

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
    from server import app as inference_app
    client = TestClient(inference_app)

    payload = {
        "intent": "summarize.doc",
        "prompt": "hello",
        "provider": "ollama",
        "model": "llama3.2"
    }

    # Missing scope -> 403
    r_forbidden = client.post("/inference/request", json=payload, headers={"Authorization": "Bearer none"})
    assert r_forbidden.status_code == 403

    # Valid read -> 200
    r_ok = client.post("/inference/request", json=payload, headers={"Authorization": "Bearer valid-read"})
    # Provider may not be available; we just assert it isn't 403 because consent passed.
    assert r_ok.status_code in (200, 500)
