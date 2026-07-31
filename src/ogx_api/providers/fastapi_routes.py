# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""FastAPI router for the Providers API.

This module defines the FastAPI router for the Providers API using standard
FastAPI route decorators.
"""

from typing import Annotated

from fastapi import APIRouter, Depends

from ogx_api.router_utils import create_path_dependency, standard_responses
from ogx_api.version import OGX_API_V1

from .api import Providers
from .models import InspectProviderRequest, ListProvidersResponse, ProviderInfo

# Path parameter dependencies for single-field models
get_inspect_provider_request = create_path_dependency(InspectProviderRequest)


def create_router(impl: Providers) -> APIRouter:
    """Create a FastAPI router for the Providers API."""
    router = APIRouter(
        prefix=f"/{OGX_API_V1}",
        tags=["Providers"],
        responses=standard_responses,
    )

    @router.get(
        "/providers",
        response_model=ListProvidersResponse,
        summary="List providers.",
        description="List all available providers.",
        responses={200: {"description": "A ListProvidersResponse containing information about all providers."}},
    )
    async def list_providers() -> ListProvidersResponse:
        return await impl.list_providers()

    @router.get(
        "/providers/{provider_id}",
        response_model=ProviderInfo,
        summary="Get provider.",
        description="Get detailed information about a specific provider.",
        responses={200: {"description": "A ProviderInfo object containing the provider's details."}},
    )
    async def inspect_provider(
        request: Annotated[InspectProviderRequest, Depends(get_inspect_provider_request)],
    ) -> ProviderInfo:
        return await impl.inspect_provider(request)

    return router
