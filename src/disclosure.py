"""Remote inference disclosure enforcement and untrusted-content handling."""
from __future__ import annotations

from typing import Any


LOCAL_PROVIDERS = {"ollama", "local", "on-device"}
PERMITTED_OUTCOMES = {"allow", "minimize", "redact"}


def enforce_disclosure(provider: str, body: dict[str, Any]) -> dict[str, Any]:
    if provider in LOCAL_PROVIDERS:
        return body
    decision = body.get("trust_decision")
    if not isinstance(decision, dict) or decision.get("outcome") not in PERMITTED_OUTCOMES:
        raise PermissionError("remote inference requires an allowing disclosure decision")
    allowed = set(decision.get("disclosed_fields") or ("intent", "prompt", "messages", "attachments", "tools"))
    sanitized = dict(body)
    for field in ("prompt", "messages", "attachments", "tools"):
        if field not in allowed:
            sanitized.pop(field, None)
    sanitized.pop("credentials", None)
    sanitized.pop("secrets", None)
    sanitized["provenance"] = list(body.get("provenance", []))
    sanitized["taint"] = "untrusted" if body.get("untrusted_input") else "person-authored"
    return sanitized


def treat_tool_output_as_untrusted(output: Any, source: str) -> dict[str, Any]:
    return {"content": output, "provenance": [source], "taint": "untrusted", "authorizes_actions": False}
