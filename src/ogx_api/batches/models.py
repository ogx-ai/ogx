# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Pydantic models for Batches API requests and responses.

This module defines the request and response models for the Batches API
using Pydantic with Field descriptions for OpenAPI schema generation.
"""

from typing import Literal

from openai.types import Batch as BatchObject
from pydantic import BaseModel, Field

from ogx_api.schema_utils import json_schema_type


@json_schema_type
class CreateBatchRequest(BaseModel):
    """Request model for creating a batch."""

    input_file_id: str = Field(..., description="The ID of an uploaded file containing requests for the batch.")
    endpoint: str = Field(..., description="The endpoint to be used for all requests in the batch.")
    completion_window: Literal["24h"] = Field(
        ..., description="The time window within which the batch should be processed."
    )
    metadata: dict[str, str] | None = Field(default=None, description="Optional metadata for the batch.")
    idempotency_key: str | None = Field(
        default=None, description="Optional idempotency key. When provided, enables idempotent behavior."
    )


@json_schema_type
class ListBatchesRequest(BaseModel):
    """Request model for listing batches."""

    after: str | None = Field(
        default=None, description="Optional cursor for pagination. Returns batches after this ID."
    )
    limit: int = Field(default=20, description="Maximum number of batches to return. Defaults to 20.")


@json_schema_type
class RetrieveBatchRequest(BaseModel):
    """Request model for retrieving a batch."""

    batch_id: str = Field(..., description="The ID of the batch to retrieve.")


@json_schema_type
class CancelBatchRequest(BaseModel):
    """Request model for canceling a batch."""

    batch_id: str = Field(..., description="The ID of the batch to cancel.")


@json_schema_type
class ListBatchesResponse(BaseModel):
    """Response containing a list of batch objects."""

    object: Literal["list"] = "list"
    data: list[BatchObject] = Field(..., description="List of batch objects")
    first_id: str | None = Field(default=None, description="ID of the first batch in the list")
    last_id: str | None = Field(default=None, description="ID of the last batch in the list")
    has_more: bool = Field(default=False, description="Whether there are more batches available")


__all__ = [
    "CreateBatchRequest",
    "ListBatchesRequest",
    "RetrieveBatchRequest",
    "CancelBatchRequest",
    "ListBatchesResponse",
    "BatchObject",
]
