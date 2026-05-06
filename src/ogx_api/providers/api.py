# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Protocol, runtime_checkable

from .models import InspectProviderRequest, ListProvidersResponse, ProviderInfo


@runtime_checkable
class Providers(Protocol):
    """Protocol for listing and inspecting providers."""

    async def list_providers(self) -> ListProvidersResponse: ...

    async def inspect_provider(self, request: InspectProviderRequest) -> ProviderInfo: ...
