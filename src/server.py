from fastapi import FastAPI, Request, Body, HTTPException, Depends
import uvicorn
import logging
import json
import time
import os
from typing import Any, Dict, List, Optional, Tuple
from unison_common.logging import configure_logging, log_json
from unison_common.tracing_middleware import TracingMiddleware
from unison_common.tracing import get_tracer, initialize_tracing, instrument_fastapi, instrument_httpx
from unison_common.consent import require_consent, ConsentScopes
try:
    from unison_common import BatonMiddleware
except Exception:
    BatonMiddleware = None
from collections import defaultdict

try:
    from .settings import InferenceServiceSettings
except ImportError:  # pragma: no cover - fallback for direct script execution
    from settings import InferenceServiceSettings  # type: ignore

app = FastAPI(title="unison-inference")
app.add_middleware(TracingMiddleware, service_name="unison-inference")
if BatonMiddleware:
    app.add_middleware(BatonMiddleware)

logger = configure_logging("unison-inference")

# P0.3: Initialize tracing and instrument FastAPI/httpx
initialize_tracing()
instrument_fastapi(app)
instrument_httpx()

# Simple in-memory metrics
_metrics = defaultdict(int)
_start_time = time.time()


def _provider_ready_status(provider: str, model: Optional[str] = None) -> Tuple[bool, str]:
    """Lightweight readiness check for configured provider/model."""
    if provider == "ollama":
        try:
            import httpx
            with httpx.Client(timeout=2.0) as client:
                r = client.get(f"{SETTINGS.ollama.base_url}/api/tags")
            if r.status_code != 200:
                return False, "ollama unreachable"
            tags = r.json().get("models", []) if r.headers.get("content-type", "").startswith("application/json") else []
            if model:
                models = [m.get("name") for m in tags if isinstance(m, dict)]
                if model not in models and not any(
                    str(model).startswith(str(m)) or str(m).startswith(str(model)) for m in models if m
                ):
                    return False, f"model {model} not pulled"
            return True, "ok"
        except Exception as exc:
            return False, f"ollama error: {exc}"
    if provider == "openai":
        return (bool(SETTINGS.openai.api_key), "api key configured" if SETTINGS.openai.api_key else "missing api key")
    if provider == "azure":
        azure = SETTINGS.azure
        return (
            bool(azure.endpoint and azure.api_key),
            "endpoint/api key configured" if azure.endpoint and azure.api_key else "missing endpoint or api key",
        )
    return False, f"unsupported provider {provider}"


def load_settings() -> InferenceServiceSettings:
    settings = InferenceServiceSettings.from_env()
    globals()["SETTINGS"] = settings
    return settings


SETTINGS = load_settings()

@app.get("/healthz")
@app.get("/health")
def health(request: Request):
    _metrics["/health"] += 1
    event_id = request.headers.get("X-Event-ID")
    log_json(logging.INFO, "health", service="unison-inference", event_id=event_id)
    return {"status": "ok", "service": "unison-inference"}

@app.get("/metrics")
def metrics():
    """Prometheus text-format metrics."""
    uptime = time.time() - _start_time
    lines = [
        "# HELP unison_inference_requests_total Total number of requests by endpoint",
        "# TYPE unison_inference_requests_total counter",
    ]
    for k, v in _metrics.items():
        lines.append(f'unison_inference_requests_total{{endpoint="{k}"}} {v}')
    lines.extend([
        "",
        "# HELP unison_inference_uptime_seconds Service uptime in seconds",
        "# TYPE unison_inference_uptime_seconds gauge",
        f"unison_inference_uptime_seconds {uptime}",
    ])
    return "\n".join(lines)

@app.get("/readyz")
@app.get("/ready")
def ready(request: Request):
    event_id = request.headers.get("X-Event-ID")
    provider = SETTINGS.default_provider
    model = SETTINGS.default_model
    provider_ready, detail = _provider_ready_status(provider, model)
    ready = provider_ready
    log_json(
        logging.INFO,
        "ready",
        service="unison-inference",
        event_id=event_id,
        provider=provider,
        model=model,
        ready=ready,
        detail=detail,
    )
    return {
        "ready": ready,
        "provider": {"name": provider, "ready": provider_ready, "detail": detail, "model": model},
        "on_device": {
            "multimodal_model": SETTINGS.on_device_multimodal_model,
            "text_model": SETTINGS.on_device_text_model,
        },
    }

@app.post("/inference/request")
def inference_request(
    request: Request,
    body: Dict[str, Any] = Body(...),
    consent=Depends(require_consent([ConsentScopes.REPLAY_READ])) if SETTINGS.require_consent else None,
):
    """
    Handle inference requests as intents.
    Supports legacy prompt-only requests and new structured companion payloads with tools/attachments.
    """
    _metrics["/inference/request"] += 1
    event_id = request.headers.get("X-Event-ID")

    intent = body.get("intent")
    prompt = body.get("prompt")
    messages = body.get("messages")
    attachments = body.get("attachments", [])
    tools = body.get("tools") or []
    tool_choice = body.get("tool_choice", "auto")
    response_format = body.get("response_format", "text-and-tools")
    allow_fallback_raw = body.get("allow_fallback", SETTINGS.allow_cloud_fallback)
    allow_fallback = (
        allow_fallback_raw
        if isinstance(allow_fallback_raw, bool)
        else str(allow_fallback_raw).lower() in {"1", "true", "yes", "on"}
    )
    provider = body.get("provider", SETTINGS.default_provider)
    model = body.get("model")
    max_tokens = body.get("max_tokens", 1000)
    temperature = body.get("temperature", 0.7)

    if not intent:
        raise HTTPException(status_code=400, detail="Missing intent")
    if not prompt and not messages:
        raise HTTPException(status_code=400, detail="Missing prompt or messages")

    # Choose model: prefer multimodal when attachments are present
    if not model:
        if attachments:
            model = SETTINGS.on_device_multimodal_model
        else:
            model = SETTINGS.on_device_text_model or SETTINGS.default_model
    provider_ready, ready_detail = _provider_ready_status(provider, model)
    fallback_used = False
    if not provider_ready and allow_fallback:
        fallback_provider = body.get("fallback_provider") or SETTINGS.fallback_provider
        fallback_model = body.get("fallback_model") or SETTINGS.fallback_model
        if fallback_provider:
            provider = fallback_provider
            model = fallback_model or model
            provider_ready, ready_detail = _provider_ready_status(provider, model)
            fallback_used = True

    if not provider_ready:
        log_json(
            logging.WARNING,
            "provider_not_ready",
            service="unison-inference",
            event_id=event_id,
            intent=intent,
            provider=provider,
            model=model,
            detail=ready_detail,
        )
        raise HTTPException(status_code=503, detail=f"Provider not ready: {ready_detail}")

    normalized_messages = _normalize_messages(prompt, messages)
    normalized_messages = _attach_media_to_messages(normalized_messages, attachments)

    log_json(
        logging.INFO,
        "inference_request",
        service="unison-inference",
        event_id=event_id,
        intent=intent,
        provider=provider,
        model=model,
        max_tokens=max_tokens,
        tools=len(tools),
        attachments=len(attachments),
        fallback_used=fallback_used,
    )

    try:
        llm_response = _call_provider(
            provider,
            model,
            normalized_messages,
            tools=tools,
            tool_choice=tool_choice,
            max_tokens=max_tokens,
            temperature=temperature,
            response_format=response_format,
        )
        content = llm_response.get("content", "")
        response = {
            "ok": True,
            "intent": intent,
            "provider": provider,
            "model": model,
            "messages": llm_response.get("messages"),
            "result": content,
            "tool_calls": llm_response.get("tool_calls", []),
            "event_id": event_id,
            "timestamp": time.time(),
            "fallback_used": fallback_used,
            "provider_ready": provider_ready,
        }
        log_json(
            logging.INFO,
            "inference_success",
            service="unison-inference",
            event_id=event_id,
            intent=intent,
            provider=provider,
            model=model,
            result_length=len(content or ""),
            tool_calls=len(response["tool_calls"]),
        )
        return response
    except Exception as e:
        log_json(logging.ERROR, "inference_error", service="unison-inference", event_id=event_id,
                 intent=intent, provider=provider, error=str(e))
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

def _normalize_messages(prompt: Optional[str], messages: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    if messages and isinstance(messages, list):
        return messages
    if prompt:
        return [{"role": "user", "content": prompt}]
    return []


def _attach_media_to_messages(messages: List[Dict[str, Any]], attachments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not attachments:
        return messages
    enriched = list(messages)
    media_descriptions = []
    for attachment in attachments:
        if not isinstance(attachment, dict):
            continue
        label = attachment.get("label") or attachment.get("type") or "attachment"
        media_descriptions.append(label)
        if "data" in attachment and attachment.get("type") == "image":
            # Ollama chat supports images in a message-level "images" array.
            if enriched:
                enriched[-1] = dict(enriched[-1])
                enriched[-1].setdefault("images", []).append(attachment["data"])
            else:
                enriched.append({"role": "user", "content": label, "images": [attachment["data"]]})
    if media_descriptions:
        enriched.append({"role": "system", "content": f"User provided attachments: {', '.join(media_descriptions)}"})
    return enriched


def _call_provider(
    provider: str,
    model: str,
    messages: List[Dict[str, Any]],
    *,
    tools: List[Dict[str, Any]],
    tool_choice: str,
    max_tokens: int,
    temperature: float,
    response_format: str,
) -> Dict[str, Any]:
    """Route to appropriate provider and normalize response."""
    if provider == "openai":
        return _call_openai(model, messages, tools, tool_choice, max_tokens, temperature, response_format)
    if provider == "ollama":
        return _call_ollama(model, messages, tools, tool_choice, max_tokens, temperature)
    if provider == "azure":
        return _call_azure_openai(model, messages, tools, tool_choice, max_tokens, temperature, response_format)
    raise ValueError(f"Unsupported provider: {provider}")


def _call_openai(
    model: str,
    messages: List[Dict[str, Any]],
    tools: List[Dict[str, Any]],
    tool_choice: str,
    max_tokens: int,
    temperature: float,
    response_format: str,
) -> Dict[str, Any]:
    """Call OpenAI API with tool-calling support."""
    import httpx

    openai_settings = SETTINGS.openai
    headers = {
        "Authorization": f"Bearer {openai_settings.api_key}",
        "Content-Type": "application/json",
    }
    tracer = get_tracer()
    if tracer:
        headers = tracer.inject_headers(headers)
    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = tool_choice
    if response_format == "json_object":
        payload["response_format"] = {"type": "json_object"}
    with httpx.Client(timeout=30.0) as client:
        resp = client.post(f"{openai_settings.base_url}/chat/completions", json=payload, headers=headers)
    resp.raise_for_status()
    result = resp.json()
    message = result["choices"][0]["message"]
    return {
        "messages": [message],
        "content": message.get("content"),
        "tool_calls": message.get("tool_calls", []),
        "usage": result.get("usage"),
        "raw": result,
    }


def _call_ollama(
    model: str,
    messages: List[Dict[str, Any]],
    tools: List[Dict[str, Any]],
    tool_choice: str,
    max_tokens: int,
    temperature: float,
) -> Dict[str, Any]:
    """Call Ollama chat API with optional tool schema."""
    import httpx

    timeout_s = float(os.getenv("UNISON_OLLAMA_TIMEOUT_S", "300"))
    headers: Dict[str, str] = {}
    tracer = get_tracer()
    if tracer:
        headers = tracer.inject_headers(headers)
    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": temperature, "num_predict": max_tokens},
    }
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = tool_choice
    with httpx.Client(timeout=timeout_s) as client:
        resp = client.post(f"{SETTINGS.ollama.base_url}/api/chat", json=payload, headers=headers)
    resp.raise_for_status()
    result = resp.json()
    message = result.get("message", {}) if isinstance(result, dict) else {}
    content = message.get("content") if isinstance(message, dict) else None
    tool_calls = message.get("tool_calls", []) if isinstance(message, dict) else []
    return {
        "messages": [message] if message else [],
        "content": content,
        "tool_calls": tool_calls,
        "raw": result,
    }


def _call_azure_openai(
    model: str,
    messages: List[Dict[str, Any]],
    tools: List[Dict[str, Any]],
    tool_choice: str,
    max_tokens: int,
    temperature: float,
    response_format: str,
) -> Dict[str, Any]:
    """Call Azure OpenAI API with tool-calling support."""
    import httpx

    azure_settings = SETTINGS.azure
    headers = {
        "api-key": azure_settings.api_key,
        "Content-Type": "application/json",
    }
    tracer = get_tracer()
    if tracer:
        headers = tracer.inject_headers(headers)
    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = tool_choice
    if response_format == "json_object":
        payload["response_format"] = {"type": "json_object"}
    url = f"{azure_settings.endpoint}/openai/deployments/{model}/chat/completions?api-version={azure_settings.api_version}"
    with httpx.Client(timeout=30.0) as client:
        resp = client.post(url, json=payload, headers=headers)
    resp.raise_for_status()
    result = resp.json()
    message = result["choices"][0]["message"]
    return {
        "messages": [message],
        "content": message.get("content"),
        "tool_calls": message.get("tool_calls", []),
        "usage": result.get("usage"),
        "raw": result,
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8087)
