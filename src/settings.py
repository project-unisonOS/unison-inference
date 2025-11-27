"""Typed configuration for the unison-inference service."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional


def _as_bool(value: Optional[str], default: bool = False) -> bool:
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class OpenAISettings:
    api_key: Optional[str] = None
    base_url: str = "https://api.openai.com/v1"


@dataclass(frozen=True)
class AzureOpenAISettings:
    endpoint: Optional[str] = None
    api_key: Optional[str] = None
    api_version: str = "2024-02-15-preview"


@dataclass(frozen=True)
class OllamaSettings:
    base_url: str = "http://ollama:11434"


@dataclass(frozen=True)
class InferenceServiceSettings:
    default_provider: str = "ollama"
    default_model: str = "qwen2.5"
    on_device_multimodal_model: str = "qwen2.5"
    on_device_text_model: str = "qwen2.5"
    allow_cloud_fallback: bool = False
    fallback_provider: Optional[str] = None
    fallback_model: Optional[str] = None
    require_consent: bool = False
    openai: OpenAISettings = field(default_factory=OpenAISettings)
    azure: AzureOpenAISettings = field(default_factory=AzureOpenAISettings)
    ollama: OllamaSettings = field(default_factory=OllamaSettings)

    @classmethod
    def from_env(cls) -> "InferenceServiceSettings":
        return cls(
            default_provider=os.getenv("UNISON_INFERENCE_PROVIDER", "ollama"),
            default_model=os.getenv("UNISON_INFERENCE_MODEL", "llama3.2"),
            on_device_multimodal_model=os.getenv("UNISON_INFERENCE_MODEL_MULTIMODAL", "llama3.2-vision"),
            on_device_text_model=os.getenv("UNISON_INFERENCE_MODEL_TEXT", "llama3.2"),
            allow_cloud_fallback=_as_bool(os.getenv("UNISON_ALLOW_CLOUD_FALLBACK"), False),
            fallback_provider=os.getenv("UNISON_FALLBACK_PROVIDER"),
            fallback_model=os.getenv("UNISON_FALLBACK_MODEL"),
            require_consent=_as_bool(os.getenv("UNISON_REQUIRE_CONSENT"), False),
            openai=OpenAISettings(
                api_key=os.getenv("OPENAI_API_KEY"),
                base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            ),
            azure=AzureOpenAISettings(
                endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview"),
            ),
            ollama=OllamaSettings(
                base_url=os.getenv("OLLAMA_BASE_URL", "http://ollama:11434"),
            ),
        )


__all__ = ["InferenceServiceSettings", "OpenAISettings", "AzureOpenAISettings", "OllamaSettings"]
