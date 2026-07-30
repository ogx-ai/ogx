# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Inspect API protocol and models.

This module contains the Inspect protocol definition.
Pydantic models are defined in ogx_api.inspect.models.
The FastAPI router is defined in ogx_api.inspect.fastapi_routes.
"""

# Import fastapi_routes for router factory access
from . import fastapi_routes

# Import protocol for re-export
from .api import Inspect

# Import models for re-export
from .models import (
    ApiFilter,
    HealthInfo,
    ListRoutesResponse,
    RouteInfo,
    VersionInfo,
)

__all__ = [
    "Inspect",
    "ApiFilter",
    "HealthInfo",
    "ListRoutesResponse",
    "RouteInfo",
    "VersionInfo",
    "fastapi_routes",
]
