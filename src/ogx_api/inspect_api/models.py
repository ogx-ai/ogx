# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Pydantic models for Inspect API requests and responses.

This module re-exports models from ogx_api.admin.models to ensure
a single source of truth and avoid type conflicts.
"""

# Import and re-export shared models from admin
from ogx_api.admin.models import (
    ApiFilter,
    HealthInfo,
    ListRoutesResponse,
    RouteInfo,
    VersionInfo,
)

__all__ = [
    "ApiFilter",
    "RouteInfo",
    "HealthInfo",
    "VersionInfo",
    "ListRoutesResponse",
]
