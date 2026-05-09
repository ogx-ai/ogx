# ogx-api

API and Provider specifications for OGX - a lightweight package with protocol definitions and provider specs.

## Overview

`ogx-api` is a minimal dependency package that contains:

- **API Protocol Definitions**: Type-safe protocol definitions for all OGX APIs (inference, agents, safety, etc.)
- **Provider Specifications**: Provider spec definitions for building custom providers
- **Data Types**: Shared data types and models used across the OGX ecosystem
- **Type Utilities**: Strong typing utilities and schema validation

## What This Package Does NOT Include

- Server implementation (see `ogx` package)
- Provider implementations (see `ogx` package)
- CLI tools (see `ogx` package)
- Runtime orchestration (see `ogx` package)

## Use Cases

This package is designed for:

1. **Third-party Provider Developers**: Build custom providers without depending on the full OGX server
2. **Client Library Authors**: Use type definitions without server dependencies
3. **Documentation Generation**: Generate API docs from protocol definitions
4. **Type Checking**: Validate implementations against the official specs

## Installation

```bash
pip install ogx-api
```

Or with uv:

```bash
uv pip install ogx-api
```

## Dependencies

Minimal dependencies:

- `openai>=2.5.0` - For OpenAI-compatible types
- `fastapi>=0.115.0,<1.0` - For FastAPI route definitions
- `pydantic>=2.11.9` - For data validation and serialization
- `jsonschema` - For JSON schema utilities
- `opentelemetry-sdk>=1.30.0` - For telemetry
- `opentelemetry-exporter-otlp-proto-http>=1.30.0` - For OTLP export

## Versioning

This package follows semantic versioning independently from the main `ogx` package:

- **Patch versions** (0.1.x): Documentation, internal improvements
- **Minor versions** (0.x.0): New APIs, backward-compatible changes
- **Major versions** (x.0.0): Breaking changes to existing APIs

The version is determined dynamically via `setuptools-scm` at build time.

## Usage Example

```python
from ogx_api import (
    Api,
    Inference,
    InlineProviderSpec,
    OpenAIChatCompletionRequestWithExtraBody,
)


# Use protocol definitions for type checking
class MyInferenceProvider(Inference):
    async def openai_chat_completion(
        self, request: OpenAIChatCompletionRequestWithExtraBody
    ):
        # Your implementation
        pass


# Define provider specifications
my_provider_spec = InlineProviderSpec(
    api=Api.inference,
    provider_type="inline::my-provider",
    pip_packages=["my-dependencies"],
    module="my_package.providers.inference",
    config_class="my_package.providers.inference.MyConfig",
)
```

## Relationship to ogx

The main `ogx` package depends on `ogx-api` and provides:

- Full server implementation
- Built-in provider implementations
- CLI tools for running and managing stacks
- Runtime provider resolution and orchestration

## Contributing

See the main [OGX repository](https://github.com/ogx-ai/ogx) for contribution guidelines.

## License

MIT License - see LICENSE file for details.

## Links

- [Main OGX Repository](https://github.com/ogx-ai/ogx)
- [Documentation](https://ogx.ai/)
- [Client Library](https://pypi.org/project/ogx-client/)
