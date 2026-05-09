# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""FastAPI router for the Connectors API.

This module defines the FastAPI router for the Connectors API using standard
FastAPI route decorators.
"""

from typing import Annotated

from fastapi import APIRouter, Depends, Path, Query

from ogx_api.router_utils import create_path_dependency, standard_responses
from ogx_api.tools import ToolDef
from ogx_api.version import OGX_API_V1BETA

from .api import Connectors
from .models import (
    Connector,
    GetConnectorRequest,
    GetConnectorToolRequest,
    ListConnectorsResponse,
    ListConnectorToolsRequest,
    ListToolsResponse,
)

# Path parameter dependencies for single-field request models
get_connector_request = create_path_dependency(GetConnectorRequest)
list_connector_tools_request = create_path_dependency(ListConnectorToolsRequest)


def create_router(impl: Connectors) -> APIRouter:
    """Create a FastAPI router for the Connectors API.

    Args:
        impl: The Connectors implementation instance

    Returns:
        APIRouter configured for the Connectors API
    """
    router = APIRouter(
        prefix=f"/{OGX_API_V1BETA}",
        tags=["Connectors"],
        responses=standard_responses,
    )

    @router.get(
        "/connectors",
        response_model=ListConnectorsResponse,
        summary="List all connectors.",
        description="List all configured connectors.",
    )
    async def list_connectors() -> ListConnectorsResponse:
        return await impl.list_connectors()

    # NOTE: Route order matters! More specific routes must come before less specific ones.
    # /connectors/{connector_id}/tools/{tool_name} must come before /connectors/{connector_id}/tools
    # /connectors/{connector_id}/tools must come before /connectors/{connector_id}

    @router.get(
        "/connectors/{connector_id}/tools/{tool_name}",
        response_model=ToolDef,
        summary="Get a tool by name from a connector.",
        description="Get a tool definition by its name from a connector.",
    )
    async def get_connector_tool(
        connector_id: Annotated[str, Path(description="Identifier for the connector")],
        tool_name: Annotated[str, Path(description="Name of the tool")],
        authorization: Annotated[str | None, Query(description="Authorization token")] = None,
    ) -> ToolDef:
        # GetConnectorToolRequest has 2 path params, so we construct it manually
        request = GetConnectorToolRequest(connector_id=connector_id, tool_name=tool_name)
        return await impl.get_connector_tool(request, authorization=authorization)

    @router.get(
        "/connectors/{connector_id}/tools",
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
        "/connectors/{connector_id}",
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
