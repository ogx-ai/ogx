# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""SystemOne judgment/decision API protocol and models.

This module contains the SystemOne protocol definition.
Pydantic models are defined in ogx_api.systemone.models.
The FastAPI router is defined in ogx_api.systemone.fastapi_routes.
"""

# Import fastapi_routes for router factory access
from . import fastapi_routes

# Import protocol for re-export
from .api import SystemOne

# Import models for re-export
from .models import (
    SystemOneChoiceAnswer,
    SystemOneChoiceQuestion,
    SystemOneNoulAnswer,
    SystemOneNoulQuestion,
    SystemOneRequest,
    SystemOneResponse,
    SystemOneScoreAnswer,
    SystemOneScoreQuestion,
    SystemOneUsage,
)

__all__ = [
    "SystemOne",
    "SystemOneChoiceAnswer",
    "SystemOneChoiceQuestion",
    "SystemOneNoulAnswer",
    "SystemOneNoulQuestion",
    "SystemOneRequest",
    "SystemOneResponse",
    "SystemOneScoreAnswer",
    "SystemOneScoreQuestion",
    "SystemOneUsage",
    "fastapi_routes",
]
