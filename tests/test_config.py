from __future__ import annotations

from src.settings import InferenceServiceSettings


def test_inference_settings_defaults(monkeypatch):
    for key in [
        "UNISON_INFERENCE_PROVIDER",
        "UNISON_INFERENCE_MODEL",
        "UNISON_REQUIRE_CONSENT",
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_API_VERSION",
        "OLLAMA_BASE_URL",
    ]:
        monkeypatch.delenv(key, raising=False)

    settings = InferenceServiceSettings.from_env()

    assert settings.default_provider == "ollama"
    assert settings.default_model == "llama3.2"
    assert settings.require_consent is False
    assert settings.openai.base_url == "https://api.openai.com/v1"
    assert settings.azure.api_version == "2024-02-15-preview"
    assert settings.ollama.base_url == "http://ollama:11434"


def test_inference_settings_env_override(monkeypatch):
    overrides = {
        "UNISON_INFERENCE_PROVIDER": "openai",
        "UNISON_INFERENCE_MODEL": "gpt-4o",
        "UNISON_REQUIRE_CONSENT": "TRUE",
        "OPENAI_API_KEY": "test-key",
        "OPENAI_BASE_URL": "https://example.com/openai",
        "AZURE_OPENAI_ENDPOINT": "https://azure.example.com",
        "AZURE_OPENAI_API_KEY": "azure-key",
        "AZURE_OPENAI_API_VERSION": "2025-01-01",
        "OLLAMA_BASE_URL": "http://localhost:11434",
    }
    for key, value in overrides.items():
        monkeypatch.setenv(key, value)

    settings = InferenceServiceSettings.from_env()

    assert settings.default_provider == "openai"
    assert settings.default_model == "gpt-4o"
    assert settings.require_consent is True
    assert settings.openai.api_key == "test-key"
    assert settings.openai.base_url == "https://example.com/openai"
    assert settings.azure.endpoint == "https://azure.example.com"
    assert settings.azure.api_key == "azure-key"
    assert settings.azure.api_version == "2025-01-01"
    assert settings.ollama.base_url == "http://localhost:11434"
