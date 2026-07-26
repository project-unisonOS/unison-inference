from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from cryptography.hazmat.primitives.serialization import load_pem_public_key
from pydantic import ValidationError
from unison_common import SignedModelManifest

try:
    from .governed_models import Ed25519ModelManifestVerifier, ModelRegistry, ModelRegistryError
except ImportError:  # pragma: no cover - direct script execution
    from governed_models import Ed25519ModelManifestVerifier, ModelRegistry, ModelRegistryError  # type: ignore


class ModelRegistryLoadError(RuntimeError):
    pass


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelRegistryLoadError(f"unable to read model registry JSON: {path}") from exc


def _load_public_keys(directory: Path) -> dict[str, Ed25519PublicKey]:
    if not directory.is_dir():
        raise ModelRegistryLoadError(f"model registry trusted-key directory is missing: {directory}")
    keys: dict[str, Ed25519PublicKey] = {}
    for path in sorted(directory.glob("*.pem")):
        try:
            key = load_pem_public_key(path.read_bytes())
        except (OSError, ValueError) as exc:
            raise ModelRegistryLoadError(f"invalid model registry public key: {path}") from exc
        if not isinstance(key, Ed25519PublicKey):
            raise ModelRegistryLoadError(f"model registry key is not Ed25519: {path}")
        keys[path.stem] = key
    if not keys:
        raise ModelRegistryLoadError("model registry has no trusted Ed25519 public keys")
    return keys


def _load_manifests(directory: Path, registry: ModelRegistry) -> None:
    if not directory.is_dir():
        raise ModelRegistryLoadError(f"model registry manifest directory is missing: {directory}")
    paths = sorted(directory.glob("*.json"))
    if not paths:
        raise ModelRegistryLoadError("model registry has no signed manifests")
    for path in paths:
        try:
            signed = SignedModelManifest.model_validate(_read_json(path))
            registry.register(signed)
        except (ValidationError, ModelRegistryError) as exc:
            raise ModelRegistryLoadError(f"model registry rejected manifest: {path}") from exc


def _load_inventory(path: Path, registry: ModelRegistry) -> None:
    data = _read_json(path)
    if not isinstance(data, dict) or set(data) != {"schema_version", "installed_artifacts", "remote_available"}:
        raise ModelRegistryLoadError("model registry inventory has an invalid shape")
    if data["schema_version"] != "model-registry-inventory.v1":
        raise ModelRegistryLoadError("unsupported model registry inventory schema")
    installed = data["installed_artifacts"]
    remote = data["remote_available"]
    if not isinstance(installed, dict) or not all(isinstance(key, str) and isinstance(value, str) for key, value in installed.items()):
        raise ModelRegistryLoadError("installed_artifacts must map model references to file paths")
    if not isinstance(remote, list) or not all(isinstance(value, str) for value in remote):
        raise ModelRegistryLoadError("remote_available must be a list of model references")

    known = set(registry.manifests)
    referenced = set(installed) | set(remote)
    unknown = sorted(referenced - known)
    if unknown:
        raise ModelRegistryLoadError(f"model registry inventory references unknown models: {unknown}")
    for model_ref in installed:
        if registry.manifests[model_ref].execution_location != "device":
            raise ModelRegistryLoadError(f"remote model is listed as an installed artifact: {model_ref}")
    for model_ref in remote:
        if registry.manifests[model_ref].execution_location != "remote":
            raise ModelRegistryLoadError(f"device model is listed as remotely available: {model_ref}")

    artifact_paths: dict[str, Path] = {}
    for model_ref, configured_path in installed.items():
        artifact_path = Path(configured_path).expanduser()
        if not artifact_path.is_absolute():
            artifact_path = path.parent / artifact_path
        artifact_path = artifact_path.resolve()
        if not artifact_path.is_file():
            raise ModelRegistryLoadError(f"model registry artifact is missing: {artifact_path}")
        artifact_paths[model_ref] = artifact_path
    try:
        registry.inventory_installed_files(artifact_paths)
    except OSError as exc:
        raise ModelRegistryLoadError("unable to hash an installed model artifact") from exc
    registry.inventory_remote(set(remote))

    invalid = sorted(
        model_ref for model_ref in referenced
        if not registry.artifact_ready(registry.manifests[model_ref])
    )
    if invalid:
        raise ModelRegistryLoadError(f"model registry artifact integrity check failed: {invalid}")


def load_production_registry(
    *, manifests_dir: Path, trusted_keys_dir: Path, inventory_file: Path,
) -> ModelRegistry:
    """Load a fully verified, public-key-only registry or fail startup closed."""
    verifier = Ed25519ModelManifestVerifier(_load_public_keys(trusted_keys_dir))
    registry = ModelRegistry(verifier)
    _load_manifests(manifests_dir, registry)
    _load_inventory(inventory_file, registry)
    return registry
