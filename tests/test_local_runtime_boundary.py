from dataclasses import replace
import httpx
import pytest
import server


@pytest.mark.parametrize("name,digest,expected", [
    ("local-small", "sha256:" + "a" * 64, True),
    ("local-small", "a" * 64, True),
    ("local-small:unreviewed", "sha256:" + "a" * 64, False),
    ("local-small", "sha256:" + "b" * 64, False),
    ("local-small", None, False),
])
def test_readiness_requires_exact_name_and_approved_digest(monkeypatch, name, digest, expected):
    monkeypatch.setattr(server, "SETTINGS", replace(server.SETTINGS, ollama=replace(server.SETTINGS.ollama, base_url="http://127.0.0.1:11434")))
    original = httpx.Client
    def handler(request):
        assert str(request.url) == "http://127.0.0.1:11434/api/tags"
        return httpx.Response(200, json={"models": [{"name": name, "digest": digest}]})
    def client(**kwargs):
        assert kwargs["trust_env"] is False
        assert kwargs["follow_redirects"] is False
        return original(transport=httpx.MockTransport(handler), **kwargs)
    monkeypatch.setattr(httpx, "Client", client)
    assert server._provider_ready_status("ollama", "local-small", "sha256:" + "a" * 64)[0] is expected
