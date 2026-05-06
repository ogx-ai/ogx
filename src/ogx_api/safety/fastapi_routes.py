# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Annotated

from fastapi import APIRouter, Body

from ogx_api.router_utils import standard_responses
from ogx_api.version import OGX_API_V1

from .api import Safety
from .datatypes import ModerationObject, RunShieldResponse
from .models import RunModerationRequest, RunShieldRequest


def create_router(impl: Safety) -> APIRouter:
    """Create a FastAPI router for the Safety API."""
    router = APIRouter(
        prefix=f"/{OGX_API_V1}",
        tags=["Safety"],
        responses=standard_responses,
    )

    @router.post(
        "/safety/run-shield",
        response_model=RunShieldResponse,
        summary="Run Shield",
        description="Run a safety shield on messages to check for policy violations.",
        responses={
            200: {"description": "The shield response indicating any violations detected."},
        },
    )
    async def run_shield(
        request: Annotated[RunShieldRequest, Body(...)],
    ) -> RunShieldResponse:
        return await impl.run_shield(request)

    @router.post(
        "/moderations",
        response_model=ModerationObject,
        summary="Create Moderation",
        description="Classifies if text inputs are potentially harmful. OpenAI-compatible endpoint.",
        responses={
            200: {"description": "The moderation results for the input."},
        },
    )
    async def run_moderation(
        request: Annotated[RunModerationRequest, Body(...)],
    ) -> ModerationObject:
        return await impl.run_moderation(request)

    return router
