# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""FastAPI router for the Batches API.

This module defines the FastAPI router for the Batches API using standard
FastAPI route decorators. The router is defined in the API package to keep
all API-related code together.
"""

from typing import Annotated

from fastapi import APIRouter, Body, Depends

from ogx_api.batches.models import (
    CancelBatchRequest,
    CreateBatchRequest,
    ListBatchesRequest,
    RetrieveBatchRequest,
)
from ogx_api.router_utils import create_path_dependency, create_query_dependency, standard_responses
from ogx_api.version import OGX_API_V1

from .api import Batches
from .models import BatchObject, ListBatchesResponse

# Automatically generate dependency functions from Pydantic models
# This ensures the models are the single source of truth for descriptions
get_retrieve_batch_request = create_path_dependency(RetrieveBatchRequest)
get_cancel_batch_request = create_path_dependency(CancelBatchRequest)


# Automatically generate dependency function from Pydantic model
# This ensures the model is the single source of truth for descriptions and defaults
get_list_batches_request = create_query_dependency(ListBatchesRequest)


def create_router(impl: Batches) -> APIRouter:
    """Create a FastAPI router for the Batches API.

    Args:
        impl: The Batches implementation instance

    Returns:
        APIRouter configured for the Batches API
    """
    router = APIRouter(
        prefix=f"/{OGX_API_V1}",
        tags=["Batches"],
        responses=standard_responses,
    )

    @router.post(
        "/batches",
        response_model=BatchObject,
        summary="Create a new batch for processing multiple API requests.",
        description="Create a new batch for processing multiple API requests.",
        responses={
            200: {"description": "The created batch object."},
            409: {"description": "Conflict: The idempotency key was previously used with different parameters."},
        },
    )
    async def create_batch(
        request: Annotated[CreateBatchRequest, Body(...)],
    ) -> BatchObject:
        return await impl.create_batch(request)

    @router.get(
        "/batches/{batch_id}",
        response_model=BatchObject,
        summary="Retrieve information about a specific batch.",
        description="Retrieve information about a specific batch.",
        responses={
            200: {"description": "The batch object."},
        },
    )
    async def retrieve_batch(
        request: Annotated[RetrieveBatchRequest, Depends(get_retrieve_batch_request)],
    ) -> BatchObject:
        return await impl.retrieve_batch(request)

    @router.post(
        "/batches/{batch_id}/cancel",
        response_model=BatchObject,
        summary="Cancel a batch that is in progress.",
        description="Cancel a batch that is in progress.",
        responses={
            200: {"description": "The updated batch object."},
        },
    )
    async def cancel_batch(
        request: Annotated[CancelBatchRequest, Depends(get_cancel_batch_request)],
    ) -> BatchObject:
        return await impl.cancel_batch(request)

    @router.get(
        "/batches",
        response_model=ListBatchesResponse,
        summary="List all batches for the current user.",
        description="List all batches for the current user.",
        responses={
            200: {"description": "A list of batch objects."},
        },
    )
    async def list_batches(
        request: Annotated[ListBatchesRequest, Depends(get_list_batches_request)],
    ) -> ListBatchesResponse:
        return await impl.list_batches(request)

    return router
