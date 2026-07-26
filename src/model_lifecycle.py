from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from pydantic import ValidationError
from unison_common import (
    GoldenSemanticJourney,
    ModelCompatibilityMatrix,
    ModelDeployment,
    ModelEvaluationResult,
    ModelHardwareQualification,
    ModelHealthSignal,
)


STATE_SCHEMA = "model-lifecycle-state.v1"
ROLLBACK_ARTIFACT_SCHEMA = "model-rollback-artifact.v1"
MAX_STATE_BYTES = 8 * 1024 * 1024
MAX_HEALTH_SIGNALS_PER_TASK = 1000


class ModelLifecycleError(ValueError):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _canonical(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


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
        model_ref=model_ref,
        journey_id=journey.journey_id,
        passed=passed,
        required_fact_diff=required_fact_diff,
        semantic_diff=semantic_diff,
        modality_equivalent=modality_equivalent,
        disclosure_fields=disclosure,
        latency_ms=int(output.get("latency_ms", 0)),
    )


def state_fingerprint(state: dict[str, Any]) -> str:
    """Fingerprint identity/memory/authority/session/profile state around model changes."""
    return hashlib.sha256(json.dumps(state, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


@dataclass
class ModelLifecycleManager:
    state_path: Path | None = None
    rollback_artifact_dir: Path | None = None
    deployments: dict[str, ModelDeployment] = field(default_factory=dict)
    candidates: dict[str, str] = field(default_factory=dict)
    evaluations: dict[str, list[ModelEvaluationResult]] = field(default_factory=dict)
    health_signals: dict[str, list[ModelHealthSignal]] = field(default_factory=dict)
    model_artifacts: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.state_path is not None:
            self.state_path = Path(self.state_path)
            self._restore()
        if self.rollback_artifact_dir is not None:
            self.rollback_artifact_dir = Path(self.rollback_artifact_dir)

    @property
    def persistent(self) -> bool:
        return self.state_path is not None

    def _state_document(self) -> dict[str, Any]:
        return {
            "schema_version": STATE_SCHEMA,
            "updated_at": _now(),
            "deployments": {
                task: deployment.model_dump(mode="json")
                for task, deployment in sorted(self.deployments.items())
            },
            "candidates": dict(sorted(self.candidates.items())),
            "evaluations": {
                model_ref: [result.model_dump(mode="json") for result in results]
                for model_ref, results in sorted(self.evaluations.items())
            },
            "health_signals": {
                task: [signal.model_dump(mode="json") for signal in signals]
                for task, signals in sorted(self.health_signals.items())
            },
            "model_artifacts": dict(sorted(self.model_artifacts.items())),
        }

    def _persist(self) -> None:
        if self.state_path is not None:
            _atomic_write(self.state_path, _canonical(self._state_document()))

    def _restore(self) -> None:
        assert self.state_path is not None
        if not self.state_path.exists():
            return
        if self.state_path.is_symlink():
            raise ModelLifecycleError("model lifecycle state must not be a symbolic link")
        try:
            if self.state_path.stat().st_size > MAX_STATE_BYTES:
                raise ModelLifecycleError("model lifecycle state exceeds the size limit")
            document = json.loads(self.state_path.read_text(encoding="utf-8"))
            if not isinstance(document, dict) or document.get("schema_version") != STATE_SCHEMA:
                raise ModelLifecycleError("unsupported model lifecycle state schema")
            expected = {
                "schema_version", "updated_at", "deployments", "candidates",
                "evaluations", "health_signals", "model_artifacts",
            }
            if set(document) != expected:
                raise ModelLifecycleError("model lifecycle state fields do not match the contract")
            deployments = {
                task: ModelDeployment.model_validate(value)
                for task, value in document["deployments"].items()
            }
            if any(task != deployment.task for task, deployment in deployments.items()):
                raise ModelLifecycleError("deployment task key does not match its record")
            candidates = {str(task): str(ref) for task, ref in document["candidates"].items()}
            if not set(candidates).issubset(deployments):
                raise ModelLifecycleError("candidate references an unknown deployment task")
            evaluations = {
                str(model_ref): [ModelEvaluationResult.model_validate(item) for item in values]
                for model_ref, values in document["evaluations"].items()
            }
            health_signals = {
                str(task): [ModelHealthSignal.model_validate(item) for item in values]
                for task, values in document["health_signals"].items()
            }
            if any(len(values) > MAX_HEALTH_SIGNALS_PER_TASK for values in health_signals.values()):
                raise ModelLifecycleError("model lifecycle health history exceeds the retention limit")
            model_artifacts = {
                str(model_ref): str(artifact_ref)
                for model_ref, artifact_ref in document["model_artifacts"].items()
            }
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError, AttributeError, ValidationError) as error:
            raise ModelLifecycleError("model lifecycle state is invalid") from error
        self.deployments = deployments
        self.candidates = candidates
        self.evaluations = evaluations
        self.health_signals = health_signals
        self.model_artifacts = model_artifacts

    def establish(self, task: str, model_ref: str, *, artifact_ref: str | None = None) -> ModelDeployment:
        deployment = ModelDeployment(
            task=task, active_model_ref=model_ref, stage="active", rollback_window_open=False,
        )
        self.deployments[task] = deployment
        self.candidates.pop(task, None)
        if artifact_ref:
            self.model_artifacts[model_ref] = artifact_ref
        self._persist()
        return deployment

    def shadow(
        self,
        *,
        task: str,
        candidate_ref: str,
        journeys: list[GoldenSemanticJourney],
        runner: Callable[[str, GoldenSemanticJourney], dict[str, Any]],
        candidate_artifact_ref: str | None = None,
    ) -> list[ModelEvaluationResult]:
        if task not in self.deployments:
            raise ModelLifecycleError("task has no established model")
        results = [
            evaluate_golden_journey(
                model_ref=candidate_ref, journey=journey, output=runner(candidate_ref, journey)
            )
            for journey in journeys
        ]
        self.evaluations[candidate_ref] = results
        self.candidates[task] = candidate_ref
        if candidate_artifact_ref:
            self.model_artifacts[candidate_ref] = candidate_artifact_ref
        current = self.deployments[task]
        self.deployments[task] = current.model_copy(update={
            "stage": "shadow",
            "audit": [*current.audit, {
                "event": "shadow-evaluated",
                "candidate": candidate_ref,
                "passed": all(item.passed for item in results),
            }],
        })
        self._persist()
        return results

    def begin_canary(
        self, *, task: str, candidate_ref: str, fraction: float = .05,
    ) -> ModelDeployment:
        if self.candidates.get(task) != candidate_ref:
            raise ModelLifecycleError("candidate does not match the evaluated shadow deployment")
        results = self.evaluations.get(candidate_ref, [])
        if not results or not all(item.passed for item in results):
            raise ModelLifecycleError("candidate has not passed every golden journey")
        current = self.deployments[task]
        deployment = current.model_copy(update={
            "active_model_ref": candidate_ref,
            "prior_model_ref": current.active_model_ref,
            "stage": "canary",
            "canary_fraction": fraction,
            "rollback_window_open": True,
            "generation": current.generation + 1,
            "audit": [*current.audit, {
                "event": "canary-started", "candidate": candidate_ref, "fraction": fraction,
            }],
        })
        self.deployments[task] = deployment
        self._persist()
        return deployment

    def observe(
        self, *, task: str, health: ModelHealthSignal, max_latency_ms: int = 5000,
    ) -> ModelDeployment:
        deployment = self.deployments[task]
        if health.model_ref != deployment.active_model_ref:
            raise ModelLifecycleError("health signal does not match active candidate")
        history = self.health_signals.setdefault(task, [])
        history.append(health)
        del history[:-MAX_HEALTH_SIGNALS_PER_TASK]
        unhealthy = (
            health.contract_success_rate < .99
            or health.semantic_success_rate < .99
            or health.error_rate > .01
            or health.fallback_rate > .05
            or health.p95_latency_ms > max_latency_ms
        )
        if unhealthy and deployment.rollback_window_open:
            return self.rollback(task=task, reason="automatic health gate")
        self._persist()
        return deployment

    def promote(self, *, task: str) -> ModelDeployment:
        current = self.deployments[task]
        if current.stage != "canary":
            raise ModelLifecycleError("only a canary can be promoted")
        promoted = current.model_copy(update={
            "stage": "active",
            "canary_fraction": 1.0,
            "audit": [*current.audit, {"event": "promoted"}],
        })
        self.deployments[task] = promoted
        self.candidates.pop(task, None)
        self._persist()
        return promoted

    def close_rollback_window(self, *, task: str) -> ModelDeployment:
        current = self.deployments[task]
        closed = current.model_copy(update={
            "prior_model_ref": None,
            "rollback_window_open": False,
            "audit": [*current.audit, {"event": "rollback-window-closed"}],
        })
        self.deployments[task] = closed
        self._persist()
        return closed

    def rollback(self, *, task: str, reason: str) -> ModelDeployment:
        current = self.deployments[task]
        if not current.prior_model_ref:
            raise ModelLifecycleError("no compatible prior model is retained")
        rolled_back = current.model_copy(update={
            "active_model_ref": current.prior_model_ref,
            "prior_model_ref": current.active_model_ref,
            "stage": "rolled-back",
            "canary_fraction": 0,
            "rollback_window_open": True,
            "generation": current.generation + 1,
            "audit": [*current.audit, {"event": "rolled-back", "reason": reason}],
        })
        self.deployments[task] = rolled_back
        self.candidates.pop(task, None)
        self._persist()
        self._write_rollback_artifact(current=current, rolled_back=rolled_back, reason=reason)
        return rolled_back

    def _write_rollback_artifact(
        self, *, current: ModelDeployment, rolled_back: ModelDeployment, reason: str,
    ) -> None:
        if self.rollback_artifact_dir is None:
            return
        artifact = {
            "schema_version": ROLLBACK_ARTIFACT_SCHEMA,
            "created_at": _now(),
            "contains_person_content": False,
            "task": rolled_back.task,
            "reason": reason,
            "generation": rolled_back.generation,
            "from_model_ref": current.active_model_ref,
            "from_release_artifact": self.model_artifacts.get(current.active_model_ref),
            "target_model_ref": rolled_back.active_model_ref,
            "target_release_artifact": self.model_artifacts.get(rolled_back.active_model_ref),
            "action": {
                "kind": "activate-retained-model",
                "model_ref": rolled_back.active_model_ref,
            },
            "deployment": rolled_back.model_dump(mode="json"),
        }
        safe_task = re.sub(r"[^a-zA-Z0-9_.-]", "_", rolled_back.task)
        target = self.rollback_artifact_dir / f"{rolled_back.generation:08d}-{safe_task}.json"
        _atomic_write(target, _canonical(artifact))


def build_compatibility_matrix(
    records: list[ModelHardwareQualification],
) -> ModelCompatibilityMatrix:
    supported = sorted({record.model_ref for record in records if record.supported})
    if supported:
        notice = "Only the listed model and hardware combinations are supported. Review each limitation."
    else:
        notice = "No model and hardware combination has complete physical-device qualification."
    return ModelCompatibilityMatrix(
        records=records, supported_model_refs=supported, truthful_notice=notice,
    )
