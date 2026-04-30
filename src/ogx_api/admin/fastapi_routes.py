# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""FastAPI router for the Admin API.

This module defines the FastAPI router for the Admin API using standard
FastAPI route decorators. The router is defined in the API package to keep
all API-related code together.
"""

from typing import Annotated

from fastapi import APIRouter, Depends, Path, Query

from ogx_api.connectors.models import (
    Connector,
    GetConnectorRequest,
    GetConnectorToolRequest,
    ListConnectorsResponse,
    ListConnectorToolsRequest,
    ListToolsResponse,
)
from ogx_api.router_utils import create_path_dependency, create_query_dependency, standard_responses
from ogx_api.tools import ToolDef
from ogx_api.version import OGX_API_V1ALPHA

from .api import Admin
from .models import (
    HealthInfo,
    InspectProviderRequest,
    ListProvidersResponse,
    ListRoutesRequest,
    ListRoutesResponse,
    ProviderInfo,
    VersionInfo,
)

# Automatically generate dependency functions from Pydantic models
get_inspect_provider_request = create_path_dependency(InspectProviderRequest)
get_list_routes_request = create_query_dependency(ListRoutesRequest)
get_connector_request = create_path_dependency(GetConnectorRequest)
list_connector_tools_request = create_path_dependency(ListConnectorToolsRequest)


def create_router(impl: Admin) -> APIRouter:
    """Create a FastAPI router for the Admin API.

    Args:
        impl: The Admin implementation instance

    Returns:
        APIRouter configured for the Admin API
    """
    router = APIRouter(
        prefix=f"/{OGX_API_V1ALPHA}",
        tags=["Admin"],
        responses=standard_responses,
    )

    @router.get(
        "/admin/providers",
        response_model=ListProvidersResponse,
        summary="List all available providers",
        description="List all available providers with their configuration and health status.",
        responses={
            200: {"description": "A list of provider information objects."},
        },
    )
    async def list_providers() -> ListProvidersResponse:
        return await impl.list_providers()

    @router.get(
        "/admin/providers/{provider_id}",
        response_model=ProviderInfo,
        summary="Get provider details",
        description="Get detailed information about a specific provider.",
        responses={
            200: {"description": "The provider information object."},
            404: {"description": "Provider not found."},
        },
    )
    async def inspect_provider(
        request: Annotated[InspectProviderRequest, Depends(get_inspect_provider_request)],
    ) -> ProviderInfo:
        return await impl.inspect_provider(request)

    @router.get(
        "/admin/inspect/routes",
        response_model=ListRoutesResponse,
        summary="List all available API routes",
        description="List all available API routes with their methods and implementing providers.",
        responses={
            200: {"description": "A list of route information objects."},
        },
    )
    async def list_routes(
        request: Annotated[ListRoutesRequest, Depends(get_list_routes_request)],
    ) -> ListRoutesResponse:
        return await impl.list_routes(request)

    @router.get(
        "/admin/health",
        response_model=HealthInfo,
        summary="Get service health status",
        description="Get the current health status of the service.",
        responses={
            200: {"description": "Health information object."},
        },
    )
    async def health() -> HealthInfo:
        return await impl.health()

    @router.get(
        "/admin/version",
        response_model=VersionInfo,
        summary="Get service version",
        description="Get the version of the service.",
        responses={
            200: {"description": "Version information object."},
        },
    )
    async def version() -> VersionInfo:
        return await impl.version()

    @router.get(
        "/admin/connectors",
        response_model=ListConnectorsResponse,
        summary="List all connectors.",
        description="List all configured connectors.",
    )
    async def list_connectors() -> ListConnectorsResponse:
        return await impl.list_connectors()

    @router.get(
        "/admin/connectors/{connector_id}/tools/{tool_name}",
        response_model=ToolDef,
        summary="Get a tool by name from a connector.",
        description="Get a tool definition by its name from a connector.",
    )
    async def get_connector_tool(
        connector_id: Annotated[str, Path(description="Identifier for the connector")],
        tool_name: Annotated[str, Path(description="Name of the tool")],
        authorization: Annotated[str | None, Query(description="Authorization token")] = None,
    ) -> ToolDef:
        request = GetConnectorToolRequest(connector_id=connector_id, tool_name=tool_name)
        return await impl.get_connector_tool(request, authorization=authorization)

    @router.get(
        "/admin/connectors/{connector_id}/tools",
        response_model=ListToolsResponse,
        summary="List tools from a connector.",
        description="List all tools available from a connector.",
    )
    async def list_connector_tools(
        request: Annotated[ListConnectorToolsRequest, Depends(list_connector_tools_request)],
        authorization: Annotated[str | None, Query(description="Authorization token")] = None,
    ) -> ListToolsResponse:
        return await impl.list_connector_tools(request, authorization=authorization)

    @router.get(
        "/admin/connectors/{connector_id}",
        response_model=Connector,
        summary="Get a connector by its ID.",
        description="Get a connector by its ID.",
    )
    async def get_connector(
        request: Annotated[GetConnectorRequest, Depends(get_connector_request)],
        authorization: Annotated[str | None, Query(description="Authorization token")] = None,
    ) -> Connector:
        return await impl.get_connector(request, authorization=authorization)

    return router
