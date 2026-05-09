# registry

Provider spec declarations. Each file defines which providers are available for a given API.

## Directory Structure

```text
registry/
  __init__.py
  batches.py           # Batch processing providers
  file_processors.py   # File processor providers
  files.py             # File storage providers
  inference.py         # Inference providers (20+ remote + 2 inline)
  responses.py         # Responses API providers (inline::builtin)
  safety.py            # Safety providers (llama-guard, bedrock, etc.)
  tool_runtime.py      # Tool runtime providers
  vector_io.py         # Vector I/O providers
```

## How It Works

Each file exports an `available_providers()` function that returns a list of `ProviderSpec` objects (`InlineProviderSpec` or `RemoteProviderSpec`). These specs declare:

- `api` -- which API the provider implements
- `provider_type` -- unique identifier (e.g., `"remote::ollama"`)
- `module` -- Python module path containing the provider implementation
- `config_class` -- fully qualified path to the provider's Pydantic config class
- `pip_packages` -- additional pip dependencies needed at runtime
- `provider_data_validator` -- optional runtime data validator for per-request credentials

## Usage

At server startup, `core/distribution.py:get_provider_registry()` calls `available_providers()` from each file and builds a mapping of `Api -> provider_type -> ProviderSpec`. The resolver uses this mapping to validate and instantiate providers from the run config.
