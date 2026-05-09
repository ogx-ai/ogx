# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Providers API protocol and models.

This module contains the Providers protocol definition.
Pydantic models are defined in ogx_api.providers.models.
The FastAPI router is defined in ogx_api.providers.fastapi_routes.
"""

# Import fastapi_routes for router factory access
from . import fastapi_routes

# Import protocol for re-export
from .api import Providers

# Import models for re-export
from .models import (
    InspectProviderRequest,
    ListProvidersResponse,
    ProviderInfo,
)

__all__ = [
    "Providers",
    "ProviderInfo",
    "ListProvidersResponse",
    "InspectProviderRequest",
    "fastapi_routes",
]
