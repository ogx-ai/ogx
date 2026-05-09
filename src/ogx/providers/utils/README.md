# utils

Shared utilities used across multiple providers.

## Directory Structure

```text
utils/
  bedrock/             # AWS Bedrock-specific utilities
  inference/           # Inference utilities (OpenAI mixin, model registry, prompt adapter)
  common/              # Common utilities shared across provider types
  datasetio/           # Dataset I/O utilities
  files/               # File handling utilities
  memory/              # Memory/vector store utilities
  responses/           # Responses API store
  safety.py            # Safety utilities
  scoring/             # Scoring utilities
  tools/               # Tool utilities (MCP client, TTL cache)
  vector_io/           # Vector I/O utilities
  __init__.py
  forward_headers.py   # Provider data header forwarding
  pagination.py        # Pagination helpers for list endpoints
  scheduler.py         # Job scheduling for async operations
```

## Key Components

### OpenAI Mixin (`inference/openai_mixin.py`)

`OpenAIMixin` is the most important shared utility. It provides a standard implementation of OpenAI-compatible endpoints (chat completion, completion, embeddings) using the `AsyncOpenAI` client. Most remote inference providers extend this mixin and only need to implement `get_base_url()` and optionally customize behavior via class attributes.

### Model Registry (`inference/model_registry.py`)

`ModelRegistryHelper` manages the mapping between OGX model identifiers and provider-specific model IDs. Providers declare their supported models as `ProviderModelEntry` objects.

### Prompt Adapter (`inference/prompt_adapter.py`)

Converts between OGX's message format and provider-specific formats. Handles image content localization, tool call formatting, and other provider-specific adaptations.

### MCP Client (`tools/mcp.py`)

Model Context Protocol client used by tool runtime providers to connect to MCP servers.

### Responses Store (`responses/responses_store.py`)

Persistent storage for OpenAI Responses API objects, used by the agents provider.
