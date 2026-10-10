# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""FastAPI router for the SystemOne judgment/decision API."""

from typing import Annotated

from fastapi import APIRouter, Body

from ogx_api.router_utils import standard_responses
from ogx_api.version import OGX_API_V1

from .api import SystemOne
from .models import SystemOneRequest, SystemOneResponse


def create_router(impl: SystemOne) -> APIRouter:
    """Create a FastAPI router for the SystemOne API.

    Mounted at /v1 (not /v1alpha) despite being a new API: the SystemOne wire format is a
    de facto standard across several backends, and existing client SDKs hardcode the
    `/v1/systemone` path with no way to point them at a different prefix.
    """
    router = APIRouter(
        prefix=f"/{OGX_API_V1}",
        tags=["SystemOne"],
        responses=standard_responses,
    )

    @router.post(
        "/systemone",
        response_model=SystemOneResponse,
        summary="Answer judgment questions about a piece of state.",
        description="Run a decision model's questions against `state` and return calibrated answers.",
    )
    async def judge(request: Annotated[SystemOneRequest, Body(...)]) -> SystemOneResponse:
        return await impl.judge(request)

    return router
