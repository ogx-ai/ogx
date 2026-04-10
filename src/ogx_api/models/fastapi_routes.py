# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""FastAPI router for the Models API.

This module defines the FastAPI router for the Models API using standard
FastAPI route decorators. Supports OpenAI, Anthropic, and Google SDK
response formats via header-based SDK detection.
"""

from datetime import UTC, datetime
from typing import Annotated

from fastapi import APIRouter, Depends, Request, Response
from fastapi.responses import JSONResponse

from ogx_api.messages.models import ANTHROPIC_VERSION
from ogx_api.router_utils import create_path_dependency, standard_responses
from ogx_api.sdk_detection import SdkType, detect_sdk
from ogx_api.version import OGX_API_V1

from .api import Models
from .models import (
    AnthropicModelInfo,
    GetModelRequest,
    GoogleModelInfo,
    Model,
    OpenAIListModelsResponse,
)

# Path parameter dependencies for single-field models
get_model_request = create_path_dependency(GetModelRequest)


def create_router(impl: Models) -> APIRouter:
    """Create a FastAPI router for the Models API.

    Args:
        impl: The Models implementation instance

    Returns:
        APIRouter configured for the Models API
    """
    router = APIRouter(
        prefix=f"/{OGX_API_V1}",
        tags=["Models"],
        responses=standard_responses,
    )

    @router.get(
        "/models",
        response_model=OpenAIListModelsResponse,
        summary="List models.",
        description="List models. Returns OpenAI, Anthropic, or Google response format based on SDK headers.",
        responses={
            200: {"description": "A list of model objects."},
        },
    )
    async def list_models(raw_request: Request) -> Response:
        sdk = detect_sdk(raw_request)
        if sdk == SdkType.ANTHROPIC:
            anthropic_result = await impl.anthropic_list_models()
            return JSONResponse(
                content=anthropic_result.model_dump(exclude_none=True),
                headers={"anthropic-version": ANTHROPIC_VERSION},
            )
        elif sdk == SdkType.GOOGLE:
            google_result = await impl.google_list_models()
            return JSONResponse(content=google_result.model_dump(exclude_none=True))

        openai_result = await impl.openai_list_models()
        return JSONResponse(content=openai_result.model_dump(exclude_none=True))

    @router.get(
        "/models/{model_id:path}",
        response_model=Model,
        summary="Get a model by its identifier.",
        description="Get a model by its identifier. Returns OpenAI, Anthropic, or Google response format based on SDK headers.",
        responses={
            200: {"description": "The model object."},
        },
    )
    async def get_model(
        raw_request: Request,
        request: Annotated[GetModelRequest, Depends(get_model_request)],
    ) -> Response:
        model = await impl.get_model(request)
        sdk = detect_sdk(raw_request)

        if sdk == SdkType.ANTHROPIC:
            anthropic_model = AnthropicModelInfo(
                id=model.identifier,
                display_name=model.identifier,
                created_at=datetime.fromtimestamp(model.created, tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            )
            return JSONResponse(
                content=anthropic_model.model_dump(exclude_none=True),
                headers={"anthropic-version": ANTHROPIC_VERSION},
            )
        elif sdk == SdkType.GOOGLE:
            google_model = GoogleModelInfo(
                name=f"models/{model.identifier}",
                display_name=model.identifier,
            )
            return JSONResponse(content=google_model.model_dump(exclude_none=True))

        return JSONResponse(content=model.model_dump(exclude_none=True))

    return router
