# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""FastAPI router for the Inspect API.

This module defines the FastAPI router for the Inspect API using standard
FastAPI route decorators.
"""

import os
from typing import Annotated

from fastapi import APIRouter, HTTPException, Query, Response

from ogx_api.router_utils import PUBLIC_ROUTE_KEY, standard_responses
from ogx_api.version import OGX_API_V1

from .api import Inspect
from .models import (
    ApiFilter,
    HealthInfo,
    ListRoutesResponse,
    VersionInfo,
)

# Mirrors the Prometheus gating in ogx.telemetry.setup_telemetry(): the scrape endpoint is
# served only when the same env var attaches a PrometheusMetricReader to the MeterProvider.
# ogx_api must not import ogx, so the check is duplicated here rather than shared.
_PROMETHEUS_ENABLED_ENV = "OGX_PROMETHEUS_ENABLED"


def _prometheus_enabled() -> bool:
    return os.environ.get(_PROMETHEUS_ENABLED_ENV, "").strip().lower() in ("1", "true", "yes", "on")


def create_router(impl: Inspect) -> APIRouter:
    """Create a FastAPI router for the Inspect API."""
    router = APIRouter(
        prefix=f"/{OGX_API_V1}",
        tags=["Inspect"],
        responses=standard_responses,
    )

    @router.get(
        "/inspect/routes",
        response_model=ListRoutesResponse,
        summary="List routes.",
        description="List all available API routes with their methods and implementing providers.",
        responses={200: {"description": "Response containing information about all available routes."}},
    )
    async def list_routes(
        api_filter: Annotated[
            ApiFilter | None,
            Query(
                description="Optional filter to control which routes are returned. Can be an API level ('v1', 'v1alpha', 'v1beta') to show non-deprecated routes at that level, or 'deprecated' to show deprecated routes across all levels. If not specified, returns all non-deprecated routes."
            ),
        ] = None,
    ) -> ListRoutesResponse:
        return await impl.list_routes(api_filter)

    @router.get(
        "/health",
        response_model=HealthInfo,
        summary="Get health status.",
        description="Get the current health status of the service.",
        responses={200: {"description": "Health information indicating if the service is operational."}},
        openapi_extra={PUBLIC_ROUTE_KEY: True},
    )
    async def health() -> HealthInfo:
        return await impl.health()

    @router.get(
        "/version",
        response_model=VersionInfo,
        summary="Get version.",
        description="Get the version of the service.",
        responses={200: {"description": "Version information containing the service version number."}},
        openapi_extra={PUBLIC_ROUTE_KEY: True},
    )
    async def version() -> VersionInfo:
        return await impl.version()

    @router.get(
        "/metrics",
        summary="Get Prometheus metrics.",
        description="Expose OTel metrics in Prometheus exposition format for scrape-based "
        "monitoring systems. Returns 404 unless the Prometheus reader is enabled via the "
        "OGX_PROMETHEUS_ENABLED environment variable.",
        response_class=Response,
        include_in_schema=False,
        openapi_extra={PUBLIC_ROUTE_KEY: True},
    )
    async def metrics() -> Response:
        if not _prometheus_enabled():
            raise HTTPException(status_code=404, detail="Prometheus metrics endpoint is not enabled")

        from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

        return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

    return router
