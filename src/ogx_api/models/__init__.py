# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Models API protocol and models.

This module contains the Models protocol definition.
Pydantic models are defined in ogx_api.models.models.
The FastAPI router is defined in ogx_api.models.fastapi_routes.
"""

# Import fastapi_routes for router factory access
from . import fastapi_routes

# Import new protocol for FastAPI router
from .api import Models

# Import models for re-export
from .models import (
    CommonModelFields,
    GetModelRequest,
    ListModelsResponse,
    Model,
    ModelInput,
    ModelType,
    OpenAIListModelsResponse,
    OpenAIModel,
    RegisterModelRequest,
    UnregisterModelRequest,
)

__all__ = [
    "CommonModelFields",
    "fastapi_routes",
    "GetModelRequest",
    "ListModelsResponse",
    "Model",
    "ModelInput",
    "Models",
    "ModelType",
    "OpenAIListModelsResponse",
    "OpenAIModel",
    "RegisterModelRequest",
    "UnregisterModelRequest",
]
