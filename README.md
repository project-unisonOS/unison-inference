# unison-inference

External inference service for Unison, providing LLM integration with multiple providers.

## Status
Core service (active) — inference gateway used by orchestrator and devstack (`8087`).

## Features

- **Multi-provider support**: OpenAI, Ollama (local), Azure OpenAI
- **Intent-driven**: Handles `inference.request` and `inference.response` intents
- **Provider abstraction**: Easy switching between providers via configuration
- **Cost-aware**: Designed to work with Policy service for cost/risk checks
- **Observability**: Structured JSON logging and Prometheus metrics

## Supported Providers

### OpenAI
- Environment: `OPENAI_API_KEY`, `OPENAI_BASE_URL` (optional)
- Models: `gpt-4`, `gpt-3.5-turbo`, etc.

### Ollama (Local)
- Environment: `OLLAMA_BASE_URL` (default: `http://ollama:11434`)
- Models: `llama3.2`, `mistral`, `codellama`, etc.
- No API keys required

### Azure OpenAI
- Environment: `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_API_VERSION`
- Models: Your Azure deployment names

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `UNISON_INFERENCE_PROVIDER` | `ollama` | Default provider (openai/ollama/azure) |
| `UNISON_INFERENCE_MODEL` | `qwen2.5` | Default model name (text-first) |
| `UNISON_INFERENCE_MODEL_MULTIMODAL` | `qwen2.5` | Preferred on-device multimodal/vision model |
| `UNISON_INFERENCE_MODEL_TEXT` | `qwen2.5` | Preferred on-device text model |
| `UNISON_ALLOW_CLOUD_FALLBACK` | `false` | Allow policy-gated fallback to cloud |
| `UNISON_FALLBACK_PROVIDER` | - | Cloud provider to use when falling back |
| `UNISON_FALLBACK_MODEL` | - | Cloud model name to use when falling back |
| `OPENAI_API_KEY` | - | OpenAI API key |
| `OPENAI_BASE_URL` | `https://api.openai.com/v1` | OpenAI base URL |
| `OLLAMA_BASE_URL` | `http://ollama:11434` | Ollama API URL |
| `AZURE_OPENAI_ENDPOINT` | - | Azure OpenAI endpoint |
| `AZURE_OPENAI_API_KEY` | - | Azure OpenAI API key |
| `AZURE_OPENAI_API_VERSION` | `2024-02-15-preview` | Azure API version |
| `UNISON_REQUIRE_GOVERNED_REGISTRY` | `false` | Fail startup unless the production governed registry is configured |
| `UNISON_MODEL_REGISTRY_MANIFESTS_DIR` | - | Directory of signed per-version model manifests |
| `UNISON_MODEL_REGISTRY_TRUSTED_KEYS_DIR` | - | Directory of trusted Ed25519 public keys named `<key_id>.pem` |
| `UNISON_MODEL_REGISTRY_INVENTORY_FILE` | - | Installed-artifact and remote-availability inventory |

Copy `.env.example` to `.env` and set provider secrets before running.

### Governed production registry

Production uses Ed25519-signed manifests so the inference runtime holds public
verification keys but no manifest-signing authority. HMAC-SHA256 remains
accepted by the contract and test signer for development compatibility; the
production loader rejects it.

Set all four governed-registry variables together. Startup fails closed when
the registry is required but absent, partially configured, has an invalid
public key or signature, references an unknown model, or contains an installed
artifact whose SHA-256 digest differs from its signed manifest. The inventory
has this strict shape; relative artifact paths resolve from the inventory file:

```json
{
  "schema_version": "model-registry-inventory.v1",
  "installed_artifacts": {
    "local-small@1": "artifacts/local-small-1.bin"
  },
  "remote_available": ["approved-remote@3"]
}
```

Each manifest file is a `signed-model-manifest.v1` envelope. Public keys are
PEM-encoded Ed25519 keys and the manifest signature is 128 lowercase hex
characters over canonical JSON (`sort_keys=True`, compact separators).

## Testing
```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -c ../constraints.txt -r requirements.txt
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OTEL_SDK_DISABLED=true python -m pytest
```

## Docs

Full docs at https://project-unisonos.github.io
- Repo roles: `unison-docs/dev/unison-repo-roles.md`
- Release/branching: `unison-docs/dev/release-and-branching.md`

## API Endpoints

### POST /inference/request
Handle inference requests as intents.

**Request:**
```json
{
  "intent": "summarize.doc",
  "prompt": "Summarize this document...",
  "provider": "ollama",
  "model": "llama3.2",
  "max_tokens": 1000,
  "temperature": 0.7
}
```

**Response:**
```json
{
  "ok": true,
  "intent": "summarize.doc",
  "provider": "ollama",
  "model": "llama3.2",
  "result": "Document summary...",
  "event_id": "uuid",
  "timestamp": 1698673200
}
```

### GET /health
Service health check.

### GET /ready
Readiness check including provider availability.

### GET /metrics
Prometheus metrics.

## Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run locally
python src/server.py

# Run with Docker
docker build -t unison-inference .
docker run -p 8087:8087 unison-inference
```

## Integration with Unison

The inference service integrates with:
- **Orchestrator**: Registers inference intents and routes requests
- **Policy**: Cost/risk evaluation for external API calls
- **Context**: Stores inference history and results
- **Storage**: Persists prompts and responses

## Example Intents

- `summarize.doc`: Summarize documents or text
- `analyze.code`: Analyze or generate code
- `translate.text`: Translate between languages
- `generate.idea`: Brainstorm ideas or suggestions
