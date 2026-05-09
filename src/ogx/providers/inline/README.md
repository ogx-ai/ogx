# inline providers

In-process provider implementations that run locally within the OGX server.

## Directory Structure

```text
inline/
  responses/           # Responses API (builtin: tool calling, multi-turn conversations)
  batches/             # Batch processing for async job execution
  inference/           # Local inference (sentence-transformers, transformers)
  ios/                 # iOS on-device inference
  safety/              # Safety checks (llama-guard, code-scanner)
  vector_io/           # Vector storage (sqlite-vec, faiss, chroma, milvus, qdrant)
  tool_runtime/        # Tool runtime (RAG context retrieval)
  files/               # File storage and management
  file_processor/      # File processing (text extraction, etc.)
  __init__.py
```

## What Makes a Provider "Inline"

Inline providers run in the same process as the server. They are declared with `InlineProviderSpec` and their `provider_type` starts with `inline::` (e.g., `inline::meta-reference`).

Their factory function is typically named `get_provider_impl()` and returns an instance implementing the relevant protocol from `ogx_api`.

## Key Inline Providers

- **`responses/builtin`** -- Implements the OpenAI Responses API. Handles tool calling loops, multi-turn conversations, and streaming.
- **`inference/sentence_transformers`** -- Runs embedding models using the sentence-transformers library.
- **`inference/transformers`** -- Runs Llama models locally using the transformers library.
- **`vector_io/sqlite_vec`** -- SQLite-based vector storage using the sqlite-vec extension.
