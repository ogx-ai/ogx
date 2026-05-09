# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Batches API protocol and models.

This module contains the Batches protocol definition.
Pydantic models are defined in ogx_api.batches.models.
The FastAPI router is defined in ogx_api.batches.fastapi_routes.
"""

from openai.types import Batch as BatchObject

# Import fastapi_routes for router factory access
from . import fastapi_routes

# Import protocol for re-export
from .api import Batches

# Import models for re-export
from .models import (
    CancelBatchRequest,
    CreateBatchRequest,
    ListBatchesRequest,
    ListBatchesResponse,
    RetrieveBatchRequest,
)

__all__ = [
    "Batches",
    "BatchObject",
    "CancelBatchRequest",
    "CreateBatchRequest",
    "ListBatchesRequest",
    "ListBatchesResponse",
    "RetrieveBatchRequest",
    "fastapi_routes",
]
