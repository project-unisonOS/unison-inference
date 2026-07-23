import pytest

from routing import route_model


CANDIDATES = [
    {"provider": "ollama", "model": "qwen-local", "location": "device", "estimated_cost": 0, "risk": "low", "disclosure": "none"},
    {"provider": "openai", "model": "remote-small", "location": "remote", "estimated_cost": 0.02, "risk": "medium", "disclosure": "content"},
]


def test_private_profile_prefers_local_and_survives_offline():
    policy = {"allowed_locations": ["device", "remote"], "cost_ceiling": 1, "max_risk": "medium", "max_disclosure": "none"}
    assert route_model(candidates=CANDIDATES, policy=policy)["provider"] == "ollama"
    assert route_model(candidates=CANDIDATES, policy=policy, offline=True)["provider"] == "ollama"


def test_cost_risk_and_disclosure_limits_fail_closed():
    with pytest.raises(PermissionError, match="no model"):
        route_model(
            candidates=[CANDIDATES[1]],
            policy={"allowed_locations": ["remote"], "cost_ceiling": 0.01, "max_risk": "low", "max_disclosure": "metadata"},
        )


def test_remote_replacement_requires_explicit_profile():
    selected = route_model(
        candidates=[CANDIDATES[1]],
        policy={"allowed_locations": ["remote"], "cost_ceiling": 0.05, "max_risk": "medium", "max_disclosure": "minimized-content"},
    )
    assert selected["model"] == "remote-small"
