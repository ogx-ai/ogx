# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Admin API protocol and models.

This module contains the Admin protocol definition.
Pydantic models are defined in ogx_api.admin.models.
The FastAPI router is defined in ogx_api.admin.fastapi_routes.
"""

# Import fastapi_routes for router factory access
from . import fastapi_routes

# Import protocol for re-export
from .api import Admin

# Import models for re-export
from .models import (
    ApiFilter,
    HealthInfo,
    InspectProviderRequest,
    ListProvidersResponse,
    ListRoutesRequest,
    ListRoutesResponse,
    ProviderInfo,
    RouteInfo,
    VersionInfo,
)

__all__ = [
    "Admin",
    "ApiFilter",
    "HealthInfo",
    "InspectProviderRequest",
    "ListProvidersResponse",
    "ListRoutesRequest",
    "ListRoutesResponse",
    "ProviderInfo",
    "RouteInfo",
    "VersionInfo",
    "fastapi_routes",
]
