# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Safety API protocol and models.

This module contains the Safety protocol definition for content moderation and safety shields.
Pydantic models are defined in ogx_api.safety.models.
The FastAPI router is defined in ogx_api.safety.fastapi_routes.
"""

from . import fastapi_routes
from .api import Safety
from .datatypes import (
    ModerationObject,
    ModerationObjectResults,
    RunShieldResponse,
    SafetyViolation,
    ShieldStore,
    ViolationLevel,
)
from .models import RunModerationRequest, RunShieldRequest

__all__ = [
    "Safety",
    "ShieldStore",
    "ModerationObject",
    "ModerationObjectResults",
    "ViolationLevel",
    "SafetyViolation",
    "RunShieldResponse",
    "RunShieldRequest",
    "RunModerationRequest",
    "fastapi_routes",
]
