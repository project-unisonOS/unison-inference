from __future__ import annotations

import hashlib
import hmac
import json
from dataclasses import dataclass, field
from typing import Any, Callable

from unison_common import (
    ModelManifest, ModelRouteDecision, ModelSemanticProposal, ModelTaskRequirement,
    SignedModelManifest,
)


class ModelRegistryError(ValueError):
    pass


class ModelProposalError(ValueError):
    pass


def _canonical(manifest: ModelManifest) -> bytes:
    return json.dumps(manifest.model_dump(mode="json"), sort_keys=True, separators=(",", ":")).encode()


class ModelManifestSigner:
    def __init__(self, keys: dict[str, bytes]):
        if not keys or any(len(value) < 32 for value in keys.values()):
            raise ValueError("model manifest signing keys must contain at least 32 bytes")
        self._keys = keys

    def sign(self, manifest: ModelManifest, key_id: str) -> SignedModelManifest:
        key = self._keys[key_id]
        signature = hmac.new(key, _canonical(manifest), hashlib.sha256).hexdigest()
        return SignedModelManifest(manifest=manifest, key_id=key_id, signature=signature)

    def verify(self, signed: SignedModelManifest) -> ModelManifest:
        key = self._keys.get(signed.key_id)
        if key is None:
            raise ModelRegistryError("model manifest signer is not trusted")
        expected = hmac.new(key, _canonical(signed.manifest), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(expected, signed.signature):
            raise ModelRegistryError("model manifest signature is invalid")
        return signed.manifest


@dataclass
class ModelRegistry:
    signer: ModelManifestSigner
    manifests: dict[str, ModelManifest] = field(default_factory=dict)
    installed: dict[str, str] = field(default_factory=dict)
    remote_available: set[str] = field(default_factory=set)

    def register(self, signed: SignedModelManifest) -> ModelManifest:
        manifest = self.signer.verify(signed)
        key = f"{manifest.model_id}@{manifest.version}"
        existing = self.manifests.get(key)
        if existing and existing != manifest:
            raise ModelRegistryError("registry drift detected for immutable model version")
        self.manifests[key] = manifest
        return manifest

    def inventory_installed(self, artifacts: dict[str, bytes]) -> None:
        self.installed = {key: "sha256:" + hashlib.sha256(value).hexdigest() for key, value in artifacts.items()}

    def inventory_remote(self, candidates: set[str]) -> None:
        self.remote_available = set(candidates)

    def artifact_ready(self, manifest: ModelManifest) -> bool:
        key = f"{manifest.model_id}@{manifest.version}"
        if manifest.execution_location == "remote":
            return key in self.remote_available
        return self.installed.get(key) == manifest.artifact_digest


def route_operation(
    *, operation_id: str, requirement: ModelTaskRequirement, registry: ModelRegistry,
    policy: dict[str, Any], hardware: dict[str, Any], offline: bool,
) -> ModelRouteDecision:
    """Apply hard eligibility before inspectable person-aligned ranking."""
    prohibited = {"popularity", "sponsorship", "provider_preference", "engagement", "affiliate_value"}
    if prohibited & set(policy.get("ranking_signals", [])):
        raise PermissionError("prohibited ranking signal")
    rejected: dict[str, list[str]] = {}
    eligible: list[ModelManifest] = []
    risk_order = {"low": 0, "medium": 1, "high": 2, "critical": 3}
    allowed_fields = set(policy.get("allowed_disclosure_fields", []))
    for manifest in registry.manifests.values():
        key = f"{manifest.model_id}@{manifest.version}"
        reasons: list[str] = []
        if requirement.task not in manifest.tasks: reasons.append("task")
        if requirement.modality not in manifest.modalities: reasons.append("modality")
        if requirement.language not in manifest.languages: reasons.append("language")
        if manifest.context_tokens < requirement.min_context_tokens: reasons.append("context")
        if requirement.structured_output and not manifest.structured_output: reasons.append("structured-output")
        if requirement.risk not in manifest.approved_risk: reasons.append("risk")
        if requirement.local_only and manifest.execution_location != "device": reasons.append("local-only")
        if offline and manifest.execution_location != "device": reasons.append("offline")
        if not manifest.license_approved: reasons.append("license")
        if policy.get("require_supported", True) and not manifest.supported: reasons.append("support")
        if manifest.estimated_cost > min(requirement.max_cost, float(policy.get("cost_ceiling", requirement.max_cost))): reasons.append("cost")
        latency = manifest.measured_latency_ms.get(requirement.task.value)
        if latency is None or latency > requirement.max_latency_ms: reasons.append("latency")
        required_fields = set(manifest.privacy.get("required_disclosure_fields", []))
        if manifest.execution_location == "remote" and not required_fields.issubset(allowed_fields): reasons.append("disclosure")
        if manifest.execution_location == "remote" and policy.get("retention") not in (None, manifest.privacy.get("retention")): reasons.append("retention")
        if not registry.artifact_ready(manifest): reasons.append("artifact-integrity-or-availability")
        hw = manifest.hardware
        if hw.architectures and hardware.get("architecture") not in hw.architectures: reasons.append("architecture")
        if int(hardware.get("ram_mb", 0)) < hw.min_ram_mb: reasons.append("ram")
        if int(hardware.get("vram_mb", 0)) < hw.min_vram_mb: reasons.append("vram")
        if hw.accelerator and hardware.get("accelerator") != hw.accelerator: reasons.append("accelerator")
        if int(hardware.get("storage_mb", 0)) < hw.storage_mb: reasons.append("storage")
        if reasons:
            rejected[key] = sorted(set(reasons))
        else:
            eligible.append(manifest)

    preference = policy.get("preference", "privacy")
    def score(item: ModelManifest):
        quality = item.measured_quality.get(requirement.task.value, 0)
        latency = item.measured_latency_ms[requirement.task.value]
        locality = 0 if item.execution_location == "device" else 1
        if preference == "quality":
            return (-quality, locality, latency, item.estimated_cost, item.model_id, item.version)
        return (locality, item.estimated_cost, -quality, latency, item.model_id, item.version)

    ranked = sorted(eligible, key=score)
    ranking = [{"candidate": f"{item.model_id}@{item.version}", "rank": index + 1, "quality": item.measured_quality.get(requirement.task.value, 0), "latency_ms": item.measured_latency_ms[requirement.task.value], "location": item.execution_location, "cost": item.estimated_cost} for index, item in enumerate(ranked)]
    if not ranked:
        return ModelRouteDecision(
            operation_id=operation_id, task=requirement.task, fallback="deterministic-or-explain-unavailable",
            rejected=rejected, explanation=["No candidate passed every hard eligibility rule"],
        )
    selected = ranked[0]
    disclosure = sorted(set(selected.privacy.get("required_disclosure_fields", [])) & allowed_fields)
    return ModelRouteDecision(
        operation_id=operation_id, task=requirement.task, selected_model_id=selected.model_id,
        selected_version=selected.version, minimized_disclosure_fields=disclosure,
        fallback="deterministic-or-next-eligible", eligible=[f"{item.model_id}@{item.version}" for item in ranked],
        rejected=rejected, ranking=ranking,
        explanation=["Hard eligibility passed", f"Ranked by person policy: {preference}"],
    )


def validate_semantic_proposal(
    *, proposal: ModelSemanticProposal, requirement: ModelTaskRequirement,
    deterministic_facts: dict[str, Any], current_source_versions: dict[str, str],
    allowed_recipients: set[str], deterministic_action_ids: set[str],
) -> dict[str, Any]:
    """Reconcile an untrusted proposal; exact and high-risk content stays deterministic."""
    if proposal.source_state_versions != current_source_versions:
        raise ModelProposalError("model proposal is stale")
    missing = set(requirement.required_fact_ids) - set(proposal.fact_claims)
    if missing:
        raise ModelProposalError(f"model proposal omitted required facts: {sorted(missing)}")
    hallucinated = set(proposal.fact_claims) - set(deterministic_facts)
    if hallucinated:
        raise ModelProposalError(f"model proposal introduced unknown facts: {sorted(hallucinated)}")
    conflicts = [key for key, value in proposal.fact_claims.items() if deterministic_facts.get(key) != value]
    if conflicts:
        raise ModelProposalError(f"model proposal conflicts with deterministic facts: {sorted(conflicts)}")
    if not set(proposal.recipients).issubset(allowed_recipients):
        raise ModelProposalError("model proposal changed or invented a recipient")
    if any(action.action_id not in deterministic_action_ids for action in proposal.actions):
        raise ModelProposalError("model proposal invented an action")
    if requirement.deterministic_fallback_required and not proposal.recovery:
        raise ModelProposalError("model proposal omitted deterministic recovery")
    deterministic_language = requirement.risk in {"high", "critical"}
    return {
        "accepted": True,
        "nodes": [node.model_dump(mode="json") for node in proposal.nodes],
        "facts": {key: deterministic_facts[key] for key in requirement.required_fact_ids},
        "action_ids": sorted(deterministic_action_ids),
        "recipients": sorted(allowed_recipients),
        "recovery": proposal.recovery,
        "language_path": "deterministic" if deterministic_language else "model-assisted",
        "model_contribution_untrusted": True,
    }
