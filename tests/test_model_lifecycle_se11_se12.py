import copy

import pytest

from model_lifecycle import ModelLifecycleError, ModelLifecycleManager, build_compatibility_matrix, state_fingerprint
from unison_common import GoldenSemanticJourney, ModelHardwareQualification, ModelHealthSignal


JOURNEY = GoldenSemanticJourney(
    journey_id="bill", required_fact_ids=["amount"], required_node_ids=["amount"],
    action_ids=["draft"], provenance_source_ids=["bill"], permitted_disclosure_fields=[],
)


def output(*, regressed=False):
    return {
        "fact_ids": [] if regressed else ["amount"], "required_node_ids": ["amount"],
        "action_ids": ["draft"], "provenance_source_ids": ["bill"], "recovery": "keep unchanged",
        "modality_equivalent": not regressed, "disclosure_fields": [], "latency_ms": 100,
    }


def test_regressed_shadow_candidate_cannot_enter_canary():
    manager = ModelLifecycleManager(); manager.establish("semantic", "stable@1")
    results = manager.shadow(task="semantic", candidate_ref="regressed@2", journeys=[JOURNEY], runner=lambda *_: output(regressed=True))
    assert not results[0].passed
    with pytest.raises(ModelLifecycleError, match="golden"):
        manager.begin_canary(task="semantic", candidate_ref="regressed@2")
    assert manager.deployments["semantic"].active_model_ref == "stable@1"


def test_unhealthy_canary_rolls_back_without_changing_person_state():
    person_state = {"identity": "p", "memory": ["m"], "permissions": ["read"], "pending_actions": ["draft"], "profile": {"output": "conversation"}}
    before = state_fingerprint(copy.deepcopy(person_state))
    manager = ModelLifecycleManager(); manager.establish("semantic", "stable@1")
    manager.shadow(task="semantic", candidate_ref="candidate@2", journeys=[JOURNEY], runner=lambda *_: output())
    manager.begin_canary(task="semantic", candidate_ref="candidate@2")
    deployment = manager.observe(task="semantic", health=ModelHealthSignal(
        model_ref="candidate@2", sample_count=100, contract_success_rate=.9,
        semantic_success_rate=.9, fallback_rate=.1, error_rate=.1, p95_latency_ms=9000,
    ))
    assert deployment.active_model_ref == "stable@1" and deployment.stage == "rolled-back"
    assert before == state_fingerprint(person_state)


def test_comparison_cannot_expand_remote_disclosure():
    manager = ModelLifecycleManager(); manager.establish("semantic", "stable@1")
    results = manager.shadow(task="semantic", candidate_ref="candidate@2", journeys=[JOURNEY], runner=lambda *_: {**output(), "disclosure_fields": ["prompt"]})
    assert not results[0].passed
    assert "disclosure:expanded" in results[0].semantic_diff


def test_prior_model_is_retained_until_rollback_window_closes():
    manager = ModelLifecycleManager(); manager.establish("semantic", "stable@1")
    manager.shadow(task="semantic", candidate_ref="candidate@2", journeys=[JOURNEY], runner=lambda *_: output())
    canary = manager.begin_canary(task="semantic", candidate_ref="candidate@2")
    assert canary.prior_model_ref == "stable@1"
    assert manager.promote(task="semantic").prior_model_ref == "stable@1"
    assert manager.close_rollback_window(task="semantic").prior_model_ref is None


def test_synthetic_load_offline_update_and_rollback_do_not_claim_support():
    record = ModelHardwareQualification(
        model_ref="candidate@2", runtime_ref="fixture@1", hardware_profile="simulated-appliance",
        evidence_kind="synthetic", processor="fixture", architecture="x86_64", ram_mb=4096,
        storage_mb=10000, latency_ms={"idle": 100, "contention": 180}, concurrent_workloads=4,
        offline_passed=True, update_passed=True, rollback_passed=True,
        semantic_quality_passed=True, safe_fallback_passed=True,
        limitations=["No physical thermals or energy evidence"], supported=False,
    )
    matrix = build_compatibility_matrix([record])
    assert matrix.supported_model_refs == []
    assert "No model" in matrix.truthful_notice
