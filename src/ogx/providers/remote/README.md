# remote providers

Remote provider adapters that connect OGX APIs to external services.

## Directory Structure

```text
remote/
  inference/           # Remote inference adapters
    anthropic/         # Anthropic Claude
    azure/             # Azure OpenAI
    bedrock/           # AWS Bedrock
    cerebras/          # Cerebras Cloud
    databricks/        # Databricks
    fireworks/         # Fireworks AI
    gemini/            # Google Gemini
    groq/              # Groq
    llama_cpp_server/  # llama.cpp server
    llama_openai_compat/ # Generic OpenAI-compatible endpoints
    nvidia/            # NVIDIA NIM
    oci/               # Oracle Cloud Infrastructure
    ollama/            # Ollama
    openai/            # OpenAI
    passthrough/       # Generic passthrough to any endpoint
    runpod/            # RunPod
    sambanova/         # SambaNova
    together/          # Together AI
    vertexai/          # Google Vertex AI
    vllm/              # vLLM
    watsonx/           # IBM WatsonX
  safety/              # Remote safety services (bedrock, nvidia, passthrough, sambanova)
  vector_io/           # Remote vector storage (chroma, elasticsearch, milvus, pgvector, qdrant, weaviate, etc.)
  agents/              # Remote agents adapters
  datasetio/           # Remote dataset services (huggingface, nvidia)
  eval/                # Remote evaluation services (nvidia)
  files/               # Remote file storage (openai, s3)
  tool_runtime/        # Remote tool runtimes (bing, brave, mcp, tavily, wolfram_alpha)
  __init__.py
```

## What Makes a Provider "Remote"

Remote providers adapt an external service to the OGX API. They are declared with `RemoteProviderSpec` and their `provider_type` starts with `remote::` (e.g., `remote::ollama`).

Their factory function is typically named `get_adapter_impl()` and returns an instance implementing the relevant protocol from `ogx_api`.

## Common Pattern

Most remote inference providers extend `OpenAIMixin` from `providers/utils/inference/openai_mixin.py`, which provides a standard implementation of OpenAI-compatible chat completion, completion, and embedding endpoints using the `AsyncOpenAI` client. The provider only needs to supply the base URL and any provider-specific configuration.
