# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""FastAPI router for the Models API.

This module defines the FastAPI router for the Models API using standard
FastAPI route decorators.
"""

from typing import Annotated

from fastapi import APIRouter, Depends

from ogx_api.router_utils import create_path_dependency, standard_responses
from ogx_api.version import OGX_API_V1

from .api import Models
from .models import (
    GetModelRequest,
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
        summary="List models using the OpenAI API.",
        description="List models using the OpenAI API.",
        responses={
            200: {"description": "A list of OpenAI model objects."},
        },
    )
    async def openai_list_models() -> OpenAIListModelsResponse:
        return await impl.openai_list_models()

    @router.get(
        "/models/{model_id:path}",
        response_model=Model,
        summary="Get a model by its identifier.",
        description="Get a model by its identifier.",
        responses={
            200: {"description": "The model object."},
        },
    )
    async def get_model(
        request: Annotated[GetModelRequest, Depends(get_model_request)],
    ) -> Model:
        return await impl.get_model(request)

    return router
