# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any, Literal

from pydantic import BaseModel, Field

from ogx_api.datatypes import HealthResponse, HealthStatus
from ogx_api.schema_utils import json_schema_type

# Valid values for the route filter parameter.
# Actual API levels: v1, v1alpha, v1beta (filters by level, excludes deprecated)
# Special filter value: "deprecated" (shows deprecated routes regardless of level)
ApiFilter = Literal["v1", "v1alpha", "v1beta", "deprecated"]


@json_schema_type
class RouteInfo(BaseModel):
    """Information about an API route including its path, method, and implementing providers.

    :param route: The API endpoint path
    :param method: HTTP method for the route
    :param provider_types: List of provider types that implement this route
    """

    route: str = Field(description="The API route path")
    method: str = Field(description="The HTTP method for the route")
    provider_types: list[str] = Field(description="List of provider types implementing this route")


@json_schema_type
class HealthInfo(BaseModel):
    """Health status information for the service.

    :param status: Current health status of the service
    """

    status: HealthStatus = Field(description="The health status of the service")


@json_schema_type
class VersionInfo(BaseModel):
    """Version information for the service.

    :param version: Version number of the service
    """

    version: str = Field(description="The version string of the service")


@json_schema_type
class ListRoutesResponse(BaseModel):
    """Response containing a list of all available API routes.

    :param data: List of available route information objects
    """

    data: list[RouteInfo] = Field(description="List of available API routes")


@json_schema_type
class ProviderInfo(BaseModel):
    """Information about a registered provider including its configuration and health status.

    :param api: The API name this provider implements
    :param provider_id: Unique identifier for the provider
    :param provider_type: The type of provider implementation
    :param config: Configuration parameters for the provider
    :param health: Current health status of the provider
    """

    api: str = Field(..., description="The API name this provider implements")
    provider_id: str = Field(..., description="Unique identifier for the provider")
    provider_type: str = Field(..., description="The type of provider implementation")
    config: dict[str, Any] = Field(..., description="Configuration parameters for the provider")
    health: HealthResponse = Field(..., description="Current health status of the provider")


@json_schema_type
class ListProvidersResponse(BaseModel):
    """Response containing a list of all available providers.

    :param data: List of provider information objects
    """

    data: list[ProviderInfo] = Field(..., description="List of provider information objects")


# Request models for FastAPI
@json_schema_type
class ListRoutesRequest(BaseModel):
    """Request to list API routes.

    :param api_filter: Optional filter to control which routes are returned
    """

    api_filter: ApiFilter | None = Field(
        default=None,
        description="Filter to control which routes are returned. Can be an API level ('v1', 'v1alpha', 'v1beta') to show non-deprecated routes at that level, or 'deprecated' to show deprecated routes across all levels. If not specified, returns all non-deprecated routes.",
    )


@json_schema_type
class InspectProviderRequest(BaseModel):
    """Request to inspect a specific provider.

    :param provider_id: The ID of the provider to inspect
    """

    provider_id: str = Field(..., description="The ID of the provider to inspect.")
