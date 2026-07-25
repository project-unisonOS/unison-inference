import hashlib

import pytest

from governed_models import (
    ModelManifestSigner, ModelProposalError, ModelRegistry, ModelRegistryError,
    route_operation, validate_semantic_proposal,
)
from unison_common import ModelManifest, ModelSemanticProposal, ModelTaskRequirement


ARTIFACT = b"local model artifact"


def manifest(**changes):
    data = {
        "model_id": "local-small", "version": "1", "artifact_digest": "sha256:" + hashlib.sha256(ARTIFACT).hexdigest(),
        "source": "publisher", "provenance": ["publisher", "evaluation:synthetic"],
        "runtime": "ollama", "runtime_version": "1", "tasks": ["interpretation", "semantic-construction"],
        "modalities": ["text"], "languages": ["en"], "context_tokens": 4096, "structured_output": True,
        "hardware": {"architectures": ["x86_64"], "min_ram_mb": 1024, "storage_mb": 100},
        "execution_location": "device", "provider": "ollama", "license": "Apache-2.0", "license_approved": True,
        "privacy": {"retention": "none", "required_disclosure_fields": []},
        "measured_quality": {"interpretation": .9, "semantic-construction": .85},
        "measured_latency_ms": {"interpretation": 100, "semantic-construction": 120},
        "approved_risk": ["low", "high"], "rollback_compatible_with": ["0"], "supported": True, "estimated_cost": 0,
    }
    data.update(changes)
    return ModelManifest.model_validate(data)


def registry_with_local():
    signer = ModelManifestSigner({"release": b"s" * 32})
    registry = ModelRegistry(signer)
    model = manifest()
    registry.register(signer.sign(model, "release"))
    registry.inventory_installed({"local-small@1": ARTIFACT})
    return signer, registry


def test_registry_fails_closed_on_forgery_drift_and_incompatible_artifact():
    signer, registry = registry_with_local()
    signed = signer.sign(manifest(version="2"), "release")
    forged = signed.model_copy(update={"signature": "0" * 64})
    with pytest.raises(ModelRegistryError, match="signature"):
        registry.register(forged)
    registry.register(signed)
    changed = signer.sign(manifest(version="2", context_tokens=8192), "release")
    with pytest.raises(ModelRegistryError, match="drift"):
        registry.register(changed)
    registry.inventory_installed({"local-small@1": b"tampered"})
    assert not registry.artifact_ready(manifest())


def test_availability_never_implies_eligibility_and_routes_each_task_independently():
    signer, registry = registry_with_local()
    remote = manifest(
        model_id="remote", artifact_digest="sha256:" + "a" * 64, execution_location="remote", provider="cloud",
        privacy={"retention": "none", "required_disclosure_fields": ["prompt"]}, estimated_cost=.01,
    )
    registry.register(signer.sign(remote, "release")); registry.inventory_remote({"remote@1"})
    requirement = ModelTaskRequirement(task="interpretation", max_cost=1, local_only=True)
    decision = route_operation(
        operation_id="intent", requirement=requirement, registry=registry,
        policy={"cost_ceiling": 1, "allowed_disclosure_fields": ["prompt"]},
        hardware={"architecture": "x86_64", "ram_mb": 4096, "storage_mb": 1000}, offline=False,
    )
    assert decision.selected_model_id == "local-small"
    assert "local-only" in decision.rejected["remote@1"]


def test_offline_hardware_cost_license_support_and_disclosure_fail_closed():
    signer = ModelManifestSigner({"release": b"s" * 32}); registry = ModelRegistry(signer)
    remote = manifest(model_id="remote", artifact_digest="sha256:" + "a" * 64, execution_location="remote", provider="cloud", license_approved=False, supported=False, estimated_cost=2, privacy={"retention": "training", "required_disclosure_fields": ["secret"]})
    registry.register(signer.sign(remote, "release")); registry.inventory_remote({"remote@1"})
    decision = route_operation(
        operation_id="o", requirement=ModelTaskRequirement(task="interpretation", local_only=False, max_cost=1),
        registry=registry, policy={"cost_ceiling": 1, "allowed_disclosure_fields": []},
        hardware={"architecture": "arm64", "ram_mb": 256, "storage_mb": 10}, offline=True,
    )
    assert decision.selected_model_id is None
    assert {"offline", "license", "support", "cost", "disclosure", "architecture", "ram", "storage"}.issubset(decision.rejected["remote@1"])


def test_popularity_and_commercial_signals_are_never_ranking_inputs():
    _, registry = registry_with_local()
    with pytest.raises(PermissionError, match="prohibited"):
        route_operation(
            operation_id="o", requirement=ModelTaskRequirement(task="interpretation", max_cost=1), registry=registry,
            policy={"cost_ceiling": 1, "ranking_signals": ["popularity"]},
            hardware={"architecture": "x86_64", "ram_mb": 4096, "storage_mb": 1000}, offline=False,
        )


def proposal(**changes):
    data = {
        "operation_id": "sem", "model_id": "local-small", "model_version": "1",
        "source_state_versions": {"bill": "7"},
        "nodes": [{"node_id": "total", "kind": "value", "label": "Total", "value": 20, "required": True, "provenance": [{"source_id": "bill", "source_type": "document"}]}],
        "fact_claims": {"amount": 20}, "recipients": ["utility"], "recovery": "Keep the bill unpaid",
        "provenance": [{"source_id": "bill", "source_type": "document"}],
    }
    data.update(changes)
    return ModelSemanticProposal.model_validate(data)


def validate(candidate):
    return validate_semantic_proposal(
        proposal=candidate,
        requirement=ModelTaskRequirement(task="semantic-construction", risk="high", required_fact_ids=["amount"], deterministic_fallback_required=True),
        deterministic_facts={"amount": 20}, current_source_versions={"bill": "7"},
        allowed_recipients={"utility"}, deterministic_action_ids=set(),
    )


def test_valid_proposal_preserves_deterministic_exact_and_high_risk_language():
    accepted = validate(proposal())
    assert accepted["facts"] == {"amount": 20}
    assert accepted["language_path"] == "deterministic"
    assert accepted["model_contribution_untrusted"]


@pytest.mark.parametrize("candidate", [
    proposal(fact_claims={}),
    proposal(fact_claims={"amount": 21}),
    proposal(fact_claims={"amount": 20, "invented": 1}),
    proposal(source_state_versions={"bill": "6"}),
    proposal(recipients=["attacker"]),
    proposal(recovery=None),
])
def test_incomplete_hallucinated_stale_adversarial_proposals_fail(candidate):
    with pytest.raises(ModelProposalError):
        validate(candidate)
