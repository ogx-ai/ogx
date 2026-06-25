# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""FastAPI router for the Inspect API.

This module defines the FastAPI router for the Inspect API using standard
FastAPI route decorators.
"""

from typing import Annotated

from fastapi import APIRouter, Query

from ogx_api.router_utils import PUBLIC_ROUTE_KEY, standard_responses
from ogx_api.version import OGX_API_V1

from .api import Inspect
from .models import (
    ApiFilter,
    HealthInfo,
    ListRoutesResponse,
    VersionInfo,
)


def create_router(impl: Inspect) -> APIRouter:
    """Create a FastAPI router for the Inspect API."""
    router = APIRouter(
        prefix=f"/{OGX_API_V1}",
        tags=["Inspect"],
        responses=standard_responses,
    )

    @router.get(
        "/inspect/routes",
        response_model=ListRoutesResponse,
        summary="List routes.",
        description="List all available API routes with their methods and implementing providers.",
        responses={200: {"description": "Response containing information about all available routes."}},
    )
    async def list_routes(
        api_filter: Annotated[
            ApiFilter | None,
            Query(
                description="Optional filter to control which routes are returned. Can be an API level ('v1', 'v1alpha', 'v1beta') to show non-deprecated routes at that level, or 'deprecated' to show deprecated routes across all levels. If not specified, returns all non-deprecated routes."
            ),
        ] = None,
    ) -> ListRoutesResponse:
        return await impl.list_routes(api_filter)

    @router.get(
        "/health",
        response_model=HealthInfo,
        summary="Get health status.",
        description="Get the current health status of the service.",
        responses={200: {"description": "Health information indicating if the service is operational."}},
        openapi_extra={PUBLIC_ROUTE_KEY: True},
    )
    async def health() -> HealthInfo:
        return await impl.health()

    @router.get(
        "/version",
        response_model=VersionInfo,
        summary="Get version.",
        description="Get the version of the service.",
        responses={200: {"description": "Version information containing the service version number."}},
        openapi_extra={PUBLIC_ROUTE_KEY: True},
    )
    async def version() -> VersionInfo:
        return await impl.version()

    return router
