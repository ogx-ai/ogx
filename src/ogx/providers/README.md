# providers

All provider implementations for OGX APIs.

## Directory Structure

```text
providers/
  inline/              # In-process providers (run locally)
  remote/              # Remote service adapters (call external APIs)
  registry/            # Provider spec declarations (available_providers())
  utils/               # Shared utilities (OpenAI mixin, model registry, etc.)
  __init__.py
```

## Provider Types

- **Inline providers** (`inline/`) run computations in-process. Examples: `meta-reference` inference, `sqlite-vec` vector storage, `llama-guard` safety.
- **Remote providers** (`remote/`) adapt external services to the OGX API. Examples: `ollama`, `openai`, `fireworks`, `vllm`, `anthropic`.

## How Providers Are Registered

Each API has a file in `registry/` (e.g., `registry/inference.py`) with an `available_providers()` function that returns a list of `ProviderSpec` objects. At startup, `core/distribution.py` calls `get_provider_registry()` to collect all specs.

## How Providers Are Instantiated

When the server starts, `core/resolver.py` reads the run config, matches provider declarations to specs from the registry, sorts by dependencies, and calls each provider module's factory function (`get_adapter_impl()` for remote, `get_provider_impl()` for inline).

## Adding a New Provider

1. Create a directory under `inline/` or `remote/` for the appropriate API.
2. Implement the API protocol from `ogx_api`.
3. Add a `ProviderSpec` entry to the corresponding file in `registry/`.
4. Run `python scripts/provider_codegen.py` to regenerate distribution configs.

See [CONTRIBUTING.md](../../../CONTRIBUTING.md) for full details.
