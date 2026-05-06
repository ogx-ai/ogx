# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""FastAPI router for the Shields API.

This module defines the FastAPI router for the Shields API using standard
FastAPI route decorators.
"""

from typing import Annotated

from fastapi import APIRouter, Body, Depends

from ogx_api.router_utils import create_path_dependency, standard_responses
from ogx_api.version import OGX_API_V1

from .api import Shields
from .models import (
    GetShieldRequest,
    ListShieldsResponse,
    RegisterShieldRequest,
    Shield,
    UnregisterShieldRequest,
)

# Automatically generate dependency functions from Pydantic models
get_get_shield_request = create_path_dependency(GetShieldRequest)
get_unregister_shield_request = create_path_dependency(UnregisterShieldRequest)


def create_router(impl: Shields) -> APIRouter:
    """Create a FastAPI router for the Shields API.

    Args:
        impl: The Shields implementation instance

    Returns:
        APIRouter configured for the Shields API
    """
    router = APIRouter(
        prefix=f"/{OGX_API_V1}",
        tags=["Shields"],
        responses=standard_responses,
    )

    @router.get(
        "/shields",
        response_model=ListShieldsResponse,
        summary="List all shields.",
        description="List all shields.",
        responses={
            200: {"description": "A ListShieldsResponse."},
        },
    )
    async def list_shields() -> ListShieldsResponse:
        return await impl.list_shields()

    @router.get(
        "/shields/{identifier:path}",
        response_model=Shield,
        summary="Get a shield by its identifier.",
        description="Get a shield by its identifier.",
        responses={
            200: {"description": "A Shield."},
        },
    )
    async def get_shield(
        request: Annotated[GetShieldRequest, Depends(get_get_shield_request)],
    ) -> Shield:
        return await impl.get_shield(request)

    @router.post(
        "/shields",
        response_model=Shield,
        summary="Register a shield.",
        description="Register a shield.",
        responses={
            200: {"description": "A Shield."},
        },
        deprecated=True,
    )
    async def register_shield(
        request: Annotated[RegisterShieldRequest, Body(...)],
    ) -> Shield:
        return await impl.register_shield(request)

    @router.delete(
        "/shields/{identifier:path}",
        summary="Unregister a shield.",
        description="Unregister a shield.",
        responses={
            200: {"description": "The shield was successfully unregistered."},
        },
        deprecated=True,
    )
    async def unregister_shield(
        request: Annotated[UnregisterShieldRequest, Depends(get_unregister_shield_request)],
    ) -> None:
        return await impl.unregister_shield(request)

    return router
