# remote inference adapters

Remote inference provider adapters that connect to external AI services.

## Directory Structure

```text
inference/
  anthropic/           # Anthropic Claude models
  azure/               # Azure OpenAI Service
  bedrock/             # AWS Bedrock
  cerebras/            # Cerebras Cloud
  databricks/          # Databricks Model Serving
  fireworks/           # Fireworks AI
  gemini/              # Google Gemini API
  groq/                # Groq LPU inference
  llama_openai_compat/ # Generic OpenAI-compatible endpoints
  nvidia/              # NVIDIA NIM
  oci/                 # Oracle Cloud Infrastructure GenAI
  ollama/              # Ollama (local model serving)
  openai/              # OpenAI API
  passthrough/         # Generic passthrough to any endpoint
  runpod/              # RunPod cloud GPU
  sambanova/           # SambaNova
  tgi/                 # HuggingFace TGI and Inference API
  together/            # Together AI
  vertexai/            # Google Vertex AI
  vllm/                # vLLM inference server
  watsonx/             # IBM WatsonX
  __init__.py
```

## Common Pattern

Most remote inference providers follow the same pattern:

1. **Config class** -- A Pydantic model with `base_url`, `api_key`, and provider-specific settings.
2. **Implementation class** -- Extends `OpenAIMixin` from `providers/utils/inference/openai_mixin.py`.
3. **Model registry** -- Declares supported models as `ProviderModelEntry` objects.
4. **Factory function** -- `get_adapter_impl()` instantiates the provider with config and dependencies.

`OpenAIMixin` provides standard implementations of `openai_chat_completion()`, `openai_completion()`, and `openai_embeddings()` using the `AsyncOpenAI` client. The provider typically just needs to implement `get_base_url()`.

## Provider Data

Some providers support per-request credentials via `provider_data_validator`. This allows API keys to be passed in request headers rather than server config, useful for multi-tenant deployments.

## Adding a New Inference Provider

1. Create a directory under `inference/` with `__init__.py`, `config.py`, and the implementation.
2. Extend `OpenAIMixin` if the service is OpenAI-compatible.
3. Add a `RemoteProviderSpec` entry to `providers/registry/inference.py`.
4. Define supported models using `ProviderModelEntry` in a model registry.
