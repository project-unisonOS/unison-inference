from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

from unison_common import (
    GoldenSemanticJourney, ModelCompatibilityMatrix, ModelDeployment,
    ModelEvaluationResult, ModelHardwareQualification, ModelHealthSignal,
)


class ModelLifecycleError(ValueError):
    pass


def evaluate_golden_journey(
    *, model_ref: str, journey: GoldenSemanticJourney, output: dict[str, Any],
) -> ModelEvaluationResult:
    facts = set(output.get("fact_ids", []))
    nodes = set(output.get("required_node_ids", []))
    actions = set(output.get("action_ids", []))
    provenance = set(output.get("provenance_source_ids", []))
    required_fact_diff = sorted(set(journey.required_fact_ids) ^ facts)
    semantic_diff = []
    for label, expected, actual in (
        ("required-node", set(journey.required_node_ids), nodes),
        ("action", set(journey.action_ids), actions),
        ("provenance", set(journey.provenance_source_ids), provenance),
    ):
        if expected != actual:
            semantic_diff.append(f"{label}:{sorted(expected ^ actual)}")
    if journey.recovery_required and not output.get("recovery"):
        semantic_diff.append("recovery:missing")
    disclosure = sorted(set(output.get("disclosure_fields", [])))
    expanded_disclosure = not set(disclosure).issubset(journey.permitted_disclosure_fields)
    if expanded_disclosure:
        semantic_diff.append("disclosure:expanded")
    modality_equivalent = bool(output.get("modality_equivalent", False))
    passed = not required_fact_diff and not semantic_diff and modality_equivalent
    return ModelEvaluationResult(
        model_ref=model_ref, journey_id=journey.journey_id, passed=passed,
        required_fact_diff=required_fact_diff, semantic_diff=semantic_diff,
        modality_equivalent=modality_equivalent, disclosure_fields=disclosure,
        latency_ms=int(output.get("latency_ms", 0)),
    )


def state_fingerprint(state: dict[str, Any]) -> str:
    """Fingerprint identity/memory/authority/session/profile state around model changes."""
    return hashlib.sha256(json.dumps(state, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


@dataclass
class ModelLifecycleManager:
    deployments: dict[str, ModelDeployment] = field(default_factory=dict)
    evaluations: dict[str, list[ModelEvaluationResult]] = field(default_factory=dict)

    def establish(self, task: str, model_ref: str) -> ModelDeployment:
        deployment = ModelDeployment(task=task, active_model_ref=model_ref, stage="active", rollback_window_open=False)
        self.deployments[task] = deployment
        return deployment

    def shadow(self, *, task: str, candidate_ref: str, journeys: list[GoldenSemanticJourney], runner: Callable[[str, GoldenSemanticJourney], dict[str, Any]]) -> list[ModelEvaluationResult]:
        if task not in self.deployments:
            raise ModelLifecycleError("task has no established model")
        results = [evaluate_golden_journey(model_ref=candidate_ref, journey=journey, output=runner(candidate_ref, journey)) for journey in journeys]
        self.evaluations[candidate_ref] = results
        current = self.deployments[task]
        self.deployments[task] = current.model_copy(update={"stage": "shadow", "audit": [*current.audit, {"event": "shadow-evaluated", "candidate": candidate_ref, "passed": all(item.passed for item in results)}]})
        return results

    def begin_canary(self, *, task: str, candidate_ref: str, fraction: float = .05) -> ModelDeployment:
        results = self.evaluations.get(candidate_ref, [])
        if not results or not all(item.passed for item in results):
            raise ModelLifecycleError("candidate has not passed every golden journey")
        current = self.deployments[task]
        deployment = current.model_copy(update={
            "active_model_ref": candidate_ref, "prior_model_ref": current.active_model_ref,
            "stage": "canary", "canary_fraction": fraction, "rollback_window_open": True,
            "generation": current.generation + 1,
            "audit": [*current.audit, {"event": "canary-started", "candidate": candidate_ref, "fraction": fraction}],
        })
        self.deployments[task] = deployment
        return deployment

    def observe(self, *, task: str, health: ModelHealthSignal, max_latency_ms: int = 5000) -> ModelDeployment:
        deployment = self.deployments[task]
        if health.model_ref != deployment.active_model_ref:
            raise ModelLifecycleError("health signal does not match active candidate")
        unhealthy = (
            health.contract_success_rate < .99 or health.semantic_success_rate < .99
            or health.error_rate > .01 or health.fallback_rate > .05
            or health.p95_latency_ms > max_latency_ms
        )
        if unhealthy and deployment.rollback_window_open:
            return self.rollback(task=task, reason="automatic health gate")
        return deployment

    def promote(self, *, task: str) -> ModelDeployment:
        current = self.deployments[task]
        if current.stage != "canary":
            raise ModelLifecycleError("only a canary can be promoted")
        promoted = current.model_copy(update={"stage": "active", "canary_fraction": 1.0, "audit": [*current.audit, {"event": "promoted"}]})
        self.deployments[task] = promoted
        return promoted

    def close_rollback_window(self, *, task: str) -> ModelDeployment:
        current = self.deployments[task]
        closed = current.model_copy(update={"prior_model_ref": None, "rollback_window_open": False, "audit": [*current.audit, {"event": "rollback-window-closed"}]})
        self.deployments[task] = closed
        return closed

    def rollback(self, *, task: str, reason: str) -> ModelDeployment:
        current = self.deployments[task]
        if not current.prior_model_ref:
            raise ModelLifecycleError("no compatible prior model is retained")
        rolled_back = current.model_copy(update={
            "active_model_ref": current.prior_model_ref, "prior_model_ref": current.active_model_ref,
            "stage": "rolled-back", "canary_fraction": 0, "rollback_window_open": True,
            "generation": current.generation + 1,
            "audit": [*current.audit, {"event": "rolled-back", "reason": reason}],
        })
        self.deployments[task] = rolled_back
        return rolled_back


def build_compatibility_matrix(records: list[ModelHardwareQualification]) -> ModelCompatibilityMatrix:
    supported = sorted({record.model_ref for record in records if record.supported})
    if supported:
        notice = "Only the listed model and hardware combinations are supported. Review each limitation."
    else:
        notice = "No model and hardware combination has complete physical-device qualification."
    return ModelCompatibilityMatrix(records=records, supported_model_refs=supported, truthful_notice=notice)
