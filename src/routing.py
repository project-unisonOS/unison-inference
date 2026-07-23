"""Privacy, cost, risk, and availability-aware Phase 8 model routing."""
from __future__ import annotations

from typing import Any


LOCAL_PROVIDERS = {"ollama", "local", "on-device"}


def route_model(*, candidates: list[dict[str, Any]], policy: dict[str, Any], offline: bool = False) -> dict[str, Any]:
    allowed_locations = set(policy.get("allowed_locations") or ["device"])
    max_cost = float(policy.get("cost_ceiling", 0))
    risk_order = {"low": 0, "medium": 1, "high": 2, "critical": 3}
    max_risk = risk_order.get(policy.get("max_risk", "low"), 0)
    max_disclosure = policy.get("max_disclosure", "none")
    eligible = []
    for candidate in candidates:
        provider = candidate.get("provider")
        if candidate.get("location") not in allowed_locations:
            continue
        if offline and provider not in LOCAL_PROVIDERS:
            continue
        if float(candidate.get("estimated_cost", 0)) > max_cost:
            continue
        if risk_order.get(candidate.get("risk", "critical"), 3) > max_risk:
            continue
        if max_disclosure == "none" and provider not in LOCAL_PROVIDERS:
            continue
        if max_disclosure == "metadata" and candidate.get("disclosure") == "content":
            continue
        eligible.append(candidate)
    if not eligible:
        raise PermissionError("no model satisfies privacy, cost, risk, and availability policy")
    return min(
        eligible,
        key=lambda item: (
            0 if item.get("provider") in LOCAL_PROVIDERS else 1,
            float(item.get("estimated_cost", 0)),
            risk_order.get(item.get("risk", "critical"), 3),
        ),
    )
