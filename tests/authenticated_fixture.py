import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi.testclient import TestClient
from unison_common import auth

@pytest.fixture
def authenticated_inference(tmp_path, monkeypatch):
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    jwk = json.loads(jwt.algorithms.RSAAlgorithm.to_jwk(key.public_key()))
    jwk.update(kid="synthetic-test-key", alg="RS256", use="sig")
    revoked = set()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def respond(self, status, body):
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(body).encode())

        def do_GET(self):
            self.respond(200, {"keys": [jwk]})

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            try:
                claims = jwt.decode(body["token"], key.public_key(), algorithms=["RS256"], options={"verify_aud": False})
                assert claims["jti"] not in revoked
                self.respond(200, {"valid": True, "claims": claims})
            except Exception:
                self.respond(401, {"valid": False})

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    url = f"http://127.0.0.1:{server.server_port}"
    monkeypatch.setattr(auth, "AUTH_SERVICE_URL", url)
    monkeypatch.setattr(auth, "JWKS_URL", url + "/.well-known/jwks.json")
    monkeypatch.setattr(auth, "_jwks_cache", {"keys": None, "expires": 0, "etag": None})
    monkeypatch.setenv("UNISON_EVENT_GRAPH_DIR", str(tmp_path / "events"))
    monkeypatch.setenv("UNISON_TRACE_DIR", str(tmp_path / "traces"))
    monkeypatch.setenv("UNISON_PHASE1_PIPELINE", "false")
    for name in ("UNISON_PRINCIPAL_BINDING_TEST_BYPASS", "DISABLE_AUTH_FOR_TESTS", "UNISON_RENDERER_URL", "UNISON_EXPERIENCE_RENDERER_URL"):
        monkeypatch.delenv(name, raising=False)
    import server as inference_server
    app = inference_server.app

    def headers(person="alice", **overrides):
        now = int(time.time())
        claims = dict(sub=person, principal_id=person, principal_kind="person", person_id=person,
                      assistant_instance_id=f"assistant-{person}", household_id="household", membership_id=f"member-{person}",
                      key_handle=f"key-{person}", credential_namespace=person, data_namespace=person,
                      cache_namespace=person, index_namespace=person, aud=["inference"], scopes=["inference.local"], iat=now, exp=now+300,
                      jti=f"session-{person}", auth_method="test-key", assurance="high")
        claims.update(overrides)
        return {"Authorization": "Bearer " + jwt.encode(claims, key, algorithm="RS256", headers={"kid": jwk["kid"]})}

    with TestClient(app) as client:
        yield client, headers, revoked, tmp_path
    server.shutdown()
    server.server_close()
    worker.join(timeout=2)

