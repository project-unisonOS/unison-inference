from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
from unison_common import ModelManifest, SignedModelManifest

from governed_models import ModelManifestSigner, _canonical
from registry_loader import ModelRegistryLoadError, load_production_registry
from server import configure_governed_registry_from_settings
from settings import InferenceServiceSettings


def _manifest(artifact: bytes, **changes) -> ModelManifest:
    data = {
        "model_id": "local-small",
        "version": "1",
        "artifact_digest": "sha256:" + hashlib.sha256(artifact).hexdigest(),
        "source": "publisher",
        "provenance": ["publisher", "evaluation:synthetic"],
        "runtime": "ollama",
        "runtime_version": "1",
        "tasks": ["interpretation"],
        "modalities": ["text"],
        "languages": ["en"],
        "context_tokens": 4096,
        "structured_output": True,
        "hardware": {"architectures": ["x86_64"], "min_ram_mb": 1024, "storage_mb": 10},
        "execution_location": "device",
        "provider": "ollama",
        "license": "Apache-2.0",
        "license_approved": True,
        "privacy": {"retention": "none", "required_disclosure_fields": []},
        "measured_quality": {"interpretation": 0.9},
        "measured_latency_ms": {"interpretation": 100},
        "approved_risk": ["low"],
        "supported": False,
    }
    data.update(changes)
    return ModelManifest.model_validate(data)


def _fixture(root: Path) -> tuple[Path, Path, Path, Path, Ed25519PrivateKey]:
    manifests = root / "manifests"
    keys = root / "keys"
    artifacts = root / "artifacts"
    manifests.mkdir()
    keys.mkdir()
    artifacts.mkdir()
    artifact = artifacts / "local-small-1.bin"
    artifact.write_bytes(b"verified local artifact")
    private = Ed25519PrivateKey.generate()
    (keys / "release-2026.pem").write_bytes(private.public_key().public_bytes(Encoding.PEM, PublicFormat.SubjectPublicKeyInfo))
    manifest = _manifest(artifact.read_bytes())
    signed = SignedModelManifest(
        manifest=manifest,
        key_id="release-2026",
        algorithm="ed25519",
        signature=private.sign(_canonical(manifest)).hex(),
    )
    (manifests / "local-small-1.json").write_text(signed.model_dump_json(indent=2), encoding="utf-8")
    inventory = root / "inventory.json"
    inventory.write_text(json.dumps({
        "schema_version": "model-registry-inventory.v1",
        "installed_artifacts": {"local-small@1": "artifacts/local-small-1.bin"},
        "remote_available": [],
    }), encoding="utf-8")
    return manifests, keys, inventory, artifact, private


def test_loads_ed25519_registry_and_hashes_installed_artifacts(tmp_path):
    manifests, keys, inventory, _, _ = _fixture(tmp_path)
    registry = load_production_registry(
        manifests_dir=manifests,
        trusted_keys_dir=keys,
        inventory_file=inventory,
    )
    assert set(registry.manifests) == {"local-small@1"}
    assert registry.artifact_ready(registry.manifests["local-small@1"])


@pytest.mark.parametrize("failure", ["signature", "artifact", "unknown-reference"])
def test_registry_startup_fails_closed(tmp_path, failure):
    manifests, keys, inventory, artifact, private = _fixture(tmp_path)
    if failure == "signature":
        path = manifests / "local-small-1.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        data["signature"] = private.sign(b"different payload").hex()
        path.write_text(json.dumps(data), encoding="utf-8")
    elif failure == "artifact":
        artifact.write_bytes(b"tampered")
    else:
        data = json.loads(inventory.read_text(encoding="utf-8"))
        data["remote_available"] = ["unknown@1"]
        inventory.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ModelRegistryLoadError):
        load_production_registry(
            manifests_dir=manifests,
            trusted_keys_dir=keys,
            inventory_file=inventory,
        )


def test_production_loader_rejects_hmac_manifest_even_when_signature_is_valid(tmp_path):
    manifests, keys, inventory, artifact, _ = _fixture(tmp_path)
    manifest = _manifest(artifact.read_bytes())
    signer = ModelManifestSigner({"release": b"s" * 32})
    (manifests / "local-small-1.json").write_text(
        signer.sign(manifest, "release").model_dump_json(indent=2),
        encoding="utf-8",
    )
    with pytest.raises(ModelRegistryLoadError):
        load_production_registry(
            manifests_dir=manifests,
            trusted_keys_dir=keys,
            inventory_file=inventory,
        )


def test_required_or_partially_configured_registry_fails_before_serving():
    with pytest.raises(ModelRegistryLoadError, match="required but not configured"):
        configure_governed_registry_from_settings(
            InferenceServiceSettings(require_governed_registry=True)
        )
    with pytest.raises(ModelRegistryLoadError, match="configuration is incomplete"):
        configure_governed_registry_from_settings(
            InferenceServiceSettings(model_registry_manifests_dir="/registry/manifests")
        )
