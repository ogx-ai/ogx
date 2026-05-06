# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""FastAPI router for the Tools API.

This module defines the FastAPI router for the Tools API using standard
FastAPI route decorators. The router is defined in the API package to keep
all API-related code together.
"""

from typing import Annotated

from fastapi import APIRouter, Depends

from ogx_api.router_utils import create_query_dependency, standard_responses
from ogx_api.version import OGX_API_V1

from .api import ToolGroups
from .models import ListToolDefsResponse, ListToolsRequest

# Automatically generate dependency function from Pydantic model
# This ensures the model is the single source of truth for descriptions and defaults
get_list_tools_request = create_query_dependency(ListToolsRequest)


def create_router(impl: ToolGroups) -> APIRouter:
    """Create a FastAPI router for the Tools API.

    Args:
        impl: The ToolGroups implementation instance

    Returns:
        APIRouter configured for the Tools API
    """
    router = APIRouter(
        prefix=f"/{OGX_API_V1}",
        tags=["Tools"],
        responses=standard_responses,
    )

    @router.get(
        "/tools",
        response_model=ListToolDefsResponse,
        summary="List tools with optional tool group filter.",
        description="List tools with optional tool group filter.",
        responses={
            200: {"description": "A ListToolDefsResponse."},
        },
    )
    async def list_tools(
        request: Annotated[ListToolsRequest, Depends(get_list_tools_request)],
    ) -> ListToolDefsResponse:
        return await impl.list_tools(request)

    return router
