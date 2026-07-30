# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Provider-specific exception handling for test recording/replay.

Serializes provider SDK exceptions in record mode and reconstructs them in replay
mode so integration tests see the original exception types.

To add a provider:
1. Create providers/<name>.py
2. Define PROVIDER = ProviderConfig(
       name="mycloud",           # Registry key stored in recordings
       sdk_module=mycloud_sdk,   # The SDK module (``import mycloud_sdk``)
       create_error=create_error,  # (status_code, body, message) -> Exception
   )
3. Import the module below and add its PROVIDER to the build_providers call
"""

from . import ollama, openai
from ._config import ProviderConfig, _validate_provider


def build_providers(*configs: ProviderConfig) -> dict[str, ProviderConfig]:
    """Build PROVIDERS dict from registered provider configs. Validates on load."""
    result: dict[str, ProviderConfig] = {}
    for config in configs:
        _validate_provider(config)
        if config.name in result:
            raise ValueError(f"Duplicate provider name: {config.name}") from None
        result[config.name] = config
    return result


class GenericProviderError(Exception):
    """Generic provider error for replay when provider-specific type can't be reconstructed."""

    def __init__(self, status_code: int, body: dict | None = None, message: str = ""):
        super().__init__(message)
        self.status_code = status_code
        self.body = body


PROVIDERS: dict[str, ProviderConfig] = build_providers(
    openai.PROVIDER,
    ollama.PROVIDER,
)


def detect_provider(exc: object) -> str:
    """Detect the provider from an exception's module."""
    module = type(exc).__module__
    for config in PROVIDERS.values():
        if module.startswith(config._module_prefix):
            return config.name
    return "unknown"


def create_provider_error(provider: str, status_code: int, body: dict | None, message: str) -> Exception:
    """Reconstruct a provider-specific error from recorded data."""
    if provider in PROVIDERS:
        return PROVIDERS[provider].create_error(status_code, body, message)

    return GenericProviderError(status_code, body, message)
