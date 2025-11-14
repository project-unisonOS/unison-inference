from fastapi import FastAPI, Request, Body, HTTPException, Depends
import uvicorn
import logging
import json
import time
from typing import Any, Dict, List, Optional
from unison_common.logging import configure_logging, log_json
from unison_common.tracing_middleware import TracingMiddleware
from unison_common.tracing import get_tracer, initialize_tracing, instrument_fastapi, instrument_httpx
from unison_common.consent import require_consent, ConsentScopes
from collections import defaultdict

try:
    from .settings import InferenceServiceSettings
except ImportError:  # pragma: no cover - fallback for direct script execution
    from settings import InferenceServiceSettings  # type: ignore

app = FastAPI(title="unison-inference")
app.add_middleware(TracingMiddleware, service_name="unison-inference")

logger = configure_logging("unison-inference")

# P0.3: Initialize tracing and instrument FastAPI/httpx
initialize_tracing()
instrument_fastapi(app)
instrument_httpx()

# Simple in-memory metrics
_metrics = defaultdict(int)
_start_time = time.time()


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
    # Check provider availability
    provider_ready = False
    provider = SETTINGS.default_provider
    if provider == "ollama":
        # Simple check if Ollama is reachable
        try:
            import httpx
            with httpx.Client(timeout=2.0) as client:
                r = client.get(f"{SETTINGS.ollama.base_url}/api/tags")
            provider_ready = r.status_code == 200
        except Exception:
            provider_ready = False
    elif provider == "openai":
        provider_ready = bool(SETTINGS.openai.api_key)
    elif provider == "azure":
        azure = SETTINGS.azure
        provider_ready = bool(azure.endpoint and azure.api_key)
    
    ready = provider_ready
    log_json(logging.INFO, "ready", service="unison-inference", event_id=event_id, provider=provider, ready=ready)
    return {"ready": ready, "provider": {"name": provider, "ready": provider_ready}}

@app.post("/inference/request")
def inference_request(
    request: Request,
    body: Dict[str, Any] = Body(...),
    consent=Depends(require_consent([ConsentScopes.REPLAY_READ])) if SETTINGS.require_consent else None,
):
    """
    Handle inference requests as intents.
    Expected body: {
        "intent": "summarize.doc",
        "prompt": "...",
        "context": {...},
        "provider": "openai|ollama|azure" (optional),
        "model": "model-name" (optional),
        "max_tokens": 1000 (optional),
        "temperature": 0.7 (optional)
    }
    """
    _metrics["/inference/request"] += 1
    event_id = request.headers.get("X-Event-ID")
    
    intent = body.get("intent")
    prompt = body.get("prompt")
    provider = body.get("provider", SETTINGS.default_provider)
    model = body.get("model", SETTINGS.default_model)
    max_tokens = body.get("max_tokens", 1000)
    temperature = body.get("temperature", 0.7)
    
    if not intent or not prompt:
        raise HTTPException(status_code=400, detail="Missing intent or prompt")
    
    log_json(logging.INFO, "inference_request", service="unison-inference", event_id=event_id, 
             intent=intent, provider=provider, model=model, max_tokens=max_tokens)
    
    try:
        result = _call_provider(provider, model, prompt, max_tokens, temperature)
        response = {
            "ok": True,
            "intent": intent,
            "provider": provider,
            "model": model,
            "result": result,
            "event_id": event_id,
            "timestamp": time.time()
        }
        log_json(logging.INFO, "inference_success", service="unison-inference", event_id=event_id,
                 intent=intent, provider=provider, model=model, result_length=len(result))
        return response
    except Exception as e:
        log_json(logging.ERROR, "inference_error", service="unison-inference", event_id=event_id,
                 intent=intent, provider=provider, error=str(e))
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

def _call_provider(provider: str, model: str, prompt: str, max_tokens: int, temperature: float) -> str:
    """Route to appropriate provider."""
    if provider == "openai":
        return _call_openai(model, prompt, max_tokens, temperature)
    elif provider == "ollama":
        return _call_ollama(model, prompt, max_tokens, temperature)
    elif provider == "azure":
        return _call_azure_openai(model, prompt, max_tokens, temperature)
    else:
        raise ValueError(f"Unsupported provider: {provider}")

def _call_openai(model: str, prompt: str, max_tokens: int, temperature: float) -> str:
    """Call OpenAI API."""
    import httpx
    openai_settings = SETTINGS.openai
    headers = {
        "Authorization": f"Bearer {openai_settings.api_key}",
        "Content-Type": "application/json"
    }
    # Inject tracing headers
    tracer = get_tracer()
    if tracer:
        headers = tracer.inject_headers(headers)
    data = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature
    }
    with httpx.Client(timeout=30.0) as client:
        resp = client.post(f"{openai_settings.base_url}/chat/completions", json=data, headers=headers)
    resp.raise_for_status()
    result = resp.json()
    return result["choices"][0]["message"]["content"]

def _call_ollama(model: str, prompt: str, max_tokens: int, temperature: float) -> str:
    """Call Ollama API."""
    import httpx
    # Inject tracing headers
    headers: Dict[str, str] = {}
    tracer = get_tracer()
    if tracer:
        headers = tracer.inject_headers(headers)
    data = {
        "model": model,
        "prompt": prompt,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens
        }
    }
    with httpx.Client(timeout=60.0) as client:
        resp = client.post(f"{SETTINGS.ollama.base_url}/api/generate", json=data, headers=headers)
    resp.raise_for_status()
    result = resp.json()
    return result.get("response", "")

def _call_azure_openai(model: str, prompt: str, max_tokens: int, temperature: float) -> str:
    """Call Azure OpenAI API."""
    import httpx
    azure_settings = SETTINGS.azure
    headers = {
        "api-key": azure_settings.api_key,
        "Content-Type": "application/json"
    }
    # Inject tracing headers
    tracer = get_tracer()
    if tracer:
        headers = tracer.inject_headers(headers)
    data = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature
    }
    url = f"{azure_settings.endpoint}/openai/deployments/{model}/chat/completions?api-version={azure_settings.api_version}"
    with httpx.Client(timeout=30.0) as client:
        resp = client.post(url, json=data, headers=headers)
    resp.raise_for_status()
    result = resp.json()
    return result["choices"][0]["message"]["content"]

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8087)
