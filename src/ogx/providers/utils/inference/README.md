# inference utilities

Shared utilities for inference providers: the OpenAI mixin, model registry, prompt adaptation, and streaming helpers.

## Directory Structure

```text
inference/
  __init__.py
  openai_mixin.py         # OpenAIMixin: standard OpenAI-compatible endpoint implementations
  openai_compat.py        # OpenAI compatibility helpers (parameter preparation, stream options)
  model_registry.py       # ModelRegistryHelper and ProviderModelEntry for model ID mapping
  prompt_adapter.py       # Message format conversion, image localization, tool formatting
  stream_utils.py         # Streaming response helpers
  inference_store.py      # InferenceStore for persisting chat completion logs
  http_client.py          # HTTP client utilities
  models_dev_registry.py  # Shared classify_model() for embedding/rerank model classification
  server_signature.py     # Engine verification via native endpoints (/props, /info)
  anthropic_mixin.py      # AnthropicMixin: native /v1/messages passthrough shared by providers
  anthropic_translation.py # Anthropic<->OpenAI translation and the SSE passthrough stream helper
  network_config.py       # NetworkConfig model and httpx client kwargs builder
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

Providers extend this mixin and implement `get_base_url()` to point at their API. It also
provides `anthropic_messages()`/`anthropic_count_tokens()` via translation to OpenAI chat
completions, which is the default for providers without a native Messages endpoint.

## AnthropicMixin (`anthropic_mixin.py`)

Shared implementation of the native Anthropic Messages API passthrough for providers whose
server exposes its own `/v1/messages` endpoint (Anthropic, Meta AI, Ollama, vLLM, Fireworks, DeepSeek).
It extends `OpenAIMixin` and shadows the translation fallback with a direct forward of the
request body, which the translation cannot represent (extended thinking, cache control, tool
search). It also defines `AnthropicAPIError`, which carries the upstream HTTP status and
message so the `/v1/messages` router can return them instead of a generic 500.

Providers declare it before `OpenAIMixin` in the base list and configure it with
`ClassVar` class attributes (so they stay class variables rather than pydantic fields):

- `anthropic_auth_style` -- `"x-api-key"` (default) or `"bearer"`
- `anthropic_no_key_placeholder` -- value sent as the auth header when no key is configured;
  `None` omits the header for the `"bearer"` style and raises `ValueError` (with a
  provider-data hint) for the `"x-api-key"` style
- `anthropic_messages_timeout` / `anthropic_count_tokens_timeout` -- default client timeouts
  in seconds; a configured `network.timeout` takes precedence

Providers without a `/v1/messages/count_tokens` endpoint (Fireworks, DeepSeek) override
`_anthropic_count_tokens_url()` to return `None`; counting then falls through to
`OpenAIMixin`'s default, which counts by calling `anthropic_messages` with `max_tokens=1`.

DeepSeek's Anthropic surface lives at a separate host path than its OpenAI-compatible one
(config `anthropic_base_url`), so it also overrides `_get_anthropic_base_url()`; the default
builds the URL from `get_base_url()` with a trailing `/v1` stripped.

## ModelRegistryHelper (`model_registry.py`)

Maps between OGX model identifiers and provider-specific model IDs. Each provider declares its supported models as a list of `ProviderModelEntry` objects with aliases. The helper resolves user-facing model names to the provider's internal identifiers.

## Prompt Adapter (`prompt_adapter.py`)

Handles format conversion between OGX's message types and provider-specific formats. Key functions:

- `localize_image_content()` -- Downloads remote images and converts to base64
- Tool call formatting for different provider conventions

## InferenceStore (`inference_store.py`)

Persists chat completion request/response pairs to the SqlStore. Used by the inference router to enable conversation history retrieval via the Conversations API.

## Model classification (`models_dev_registry.py`)

`classify_model(identifier, provider_id)` classifies embedding and rerank models for remote adapters whose `/v1/models` response has no model task/type field (vLLM, llama.cpp servers), returning `None` when `identifier` is neither so callers can fall back to their own default classification. Only embedding classification consults the [models.dev](https://models.dev) registry, enriching `Model.metadata` with `embedding_dimension`/`context_length` when it has an entry, and falling back to a name heuristic (`"embed"` in the identifier) otherwise; models.dev has no rerank entries, so rerank classification is a name heuristic (`"rerank"` in the identifier) only.

## Server signature verification (`server_signature.py`)

`verify_llama_cpp_server(base_url)` and `verify_text_embeddings_inference_server(base_url)` confirm that a server configured as an OpenAI-compatible endpoint is actually the expected engine, by GETting an engine-specific native endpoint on the server root (llama.cpp `GET /props`, TEI `GET /info`) and checking its response shape. They raise `ValueError` on an engine mismatch and `ServerUnreachableError` (a `ValueError` subclass) when the server cannot be reached at all. Adapters call these in `initialize()` to fail fast at construction time when a different engine is running on the configured port (only warning when the server is down, matching the Ollama adapter). The non-raising `check_llama_cpp_server()` / `check_text_embeddings_inference_server()` variants return a `HealthResponse` and back the adapters' `health()` methods.
