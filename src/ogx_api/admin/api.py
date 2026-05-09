# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Protocol, runtime_checkable

from .models import (
    HealthInfo,
    InspectProviderRequest,
    ListProvidersResponse,
    ListRoutesRequest,
    ListRoutesResponse,
    ProviderInfo,
    VersionInfo,
)


@runtime_checkable
class Admin(Protocol):
    """Admin

    Admin API for stack operations only available to administrative users.
    """

    async def list_providers(self) -> ListProvidersResponse:
        """List providers.

        List all available providers.

        :returns: A ListProvidersResponse containing information about all providers.
        """
        ...

    async def inspect_provider(self, request: InspectProviderRequest) -> ProviderInfo:
        """Get provider.

        Get detailed information about a specific provider.

        :param request: Request containing the provider ID to inspect
        :returns: A ProviderInfo object containing the provider's details.
        """
        ...

    async def list_routes(self, request: ListRoutesRequest) -> ListRoutesResponse:
        """List routes.

        List all available API routes with their methods and implementing providers.

        :param request: Request containing optional filter parameters
        :returns: Response containing information about all available routes.
        """
        ...

    async def health(self) -> HealthInfo:
        """Get health status.

        Get the current health status of the service.

        :returns: Health information indicating if the service is operational.
        """
        ...

    async def version(self) -> VersionInfo:
        """Get version.

        Get the version of the service.

        :returns: Version information containing the service version number.
        """
        ...
