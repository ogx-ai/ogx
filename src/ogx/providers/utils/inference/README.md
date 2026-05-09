# inference utilities

Shared utilities for inference providers: the OpenAI mixin, model registry, prompt adaptation, and streaming helpers.

## Directory Structure

```text
inference/
  __init__.py
  openai_mixin.py      # OpenAIMixin: standard OpenAI-compatible endpoint implementations
  openai_compat.py     # OpenAI compatibility helpers (parameter preparation, stream options)
  model_registry.py    # ModelRegistryHelper and ProviderModelEntry for model ID mapping
  prompt_adapter.py    # Message format conversion, image localization, tool formatting
  embedding_mixin.py   # Embedding-specific utilities
  stream_utils.py      # Streaming response helpers
  inference_store.py   # InferenceStore for persisting chat completion logs
  http_client.py       # HTTP client utilities
```

## OpenAIMixin (`openai_mixin.py`)

The central utility class. Provides implementations of:

- `openai_chat_completion()` -- Chat completions via `AsyncOpenAI`
- `openai_completion()` -- Text completions via `AsyncOpenAI`
- `openai_embeddings()` -- Embedding generation via `AsyncOpenAI`

Customizable via class attributes:

- `overwrite_completion_id` -- Replace response IDs with internal IDs
- `download_images` -- Download and base64-encode images for providers that need it
- `supports_stream_options` -- Disable stream_options for providers that don't support it

Providers extend this mixin and implement `get_base_url()` to point at their API.

## ModelRegistryHelper (`model_registry.py`)

Maps between OGX model identifiers and provider-specific model IDs. Each provider declares its supported models as a list of `ProviderModelEntry` objects with aliases. The helper resolves user-facing model names to the provider's internal identifiers.

## Prompt Adapter (`prompt_adapter.py`)

Handles format conversion between OGX's message types and provider-specific formats. Key functions:

- `localize_image_content()` -- Downloads remote images and converts to base64
- Tool call formatting for different provider conventions

## InferenceStore (`inference_store.py`)

Persists chat completion request/response pairs to the SqlStore. Used by the inference router to enable conversation history retrieval via the Conversations API.
