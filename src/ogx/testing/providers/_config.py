# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""ProviderConfig and validation. Separate module to allow clean top-level imports."""

from collections.abc import Callable
from dataclasses import dataclass
from types import ModuleType


@dataclass
class ProviderConfig:
    """Configuration for a provider's exception handling.

    Attributes:
        name: Registry key stored in recordings and used by create_provider_error(provider=...).
        sdk_module: The provider's SDK module (e.g. ``import openai``). The module's
            ``__name__`` is used to detect which provider raised an exception.
        create_error: Reconstructs the provider's exception from recorded data.
            Signature: (status_code: int, body: dict | None, message: str) -> Exception
    """

    name: str
    sdk_module: ModuleType
    create_error: Callable[[int, dict | None, str], Exception]

    @property
    def _module_prefix(self) -> str:
        """Top-level package name from sdk_module, used by detect_provider."""
        return self.sdk_module.__name__


def _validate_provider(config: ProviderConfig) -> None:
    """Validate ProviderConfig. Raises ValueError with clear message if invalid."""
    if not isinstance(config.name, str) or not config.name:
        raise ValueError(f"ProviderConfig.name must be a non-empty str, got {config.name!r}") from None
    if not isinstance(config.sdk_module, ModuleType):
        raise ValueError(
            f"ProviderConfig.sdk_module must be a module (e.g. ``import openai``), "
            f"got {type(config.sdk_module).__name__}"
        ) from None
    if not callable(config.create_error):
        raise ValueError(
            f"ProviderConfig.create_error must be callable, got {type(config.create_error).__name__}"
        ) from None
