# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Prompts API protocol and models.

This module contains the Prompts protocol definition.
Pydantic models are defined in ogx_api.prompts.models.
The FastAPI router is defined in ogx_api.prompts.fastapi_routes.
"""

# Import fastapi_routes for router factory access
from . import fastapi_routes

# Import protocol for FastAPI router
from .api import Prompts

# Import models for re-export
from .models import (
    CreatePromptRequest,
    DeletePromptRequest,
    GetPromptRequest,
    ListPromptsResponse,
    ListPromptVersionsRequest,
    Prompt,
    SetDefaultVersionBodyRequest,
    SetDefaultVersionRequest,
    UpdatePromptBodyRequest,
    UpdatePromptRequest,
)

__all__ = [
    "CreatePromptRequest",
    "DeletePromptRequest",
    "GetPromptRequest",
    "ListPromptVersionsRequest",
    "ListPromptsResponse",
    "Prompt",
    "Prompts",
    "SetDefaultVersionBodyRequest",
    "SetDefaultVersionRequest",
    "UpdatePromptBodyRequest",
    "UpdatePromptRequest",
    "fastapi_routes",
]
