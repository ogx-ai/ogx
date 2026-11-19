# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""
FastAPI app creation for OpenAPI generation.
"""

from fastapi import FastAPI

from ogx.core.resolver import api_protocol_map
from ogx.core.server.fastapi_router_registry import build_fastapi_router


def create_ogx_app() -> FastAPI:
    """
    Create a FastAPI app that represents the OGX API.
    Each API's router is built from the provider registry and included
    for route-based OpenAPI generation.
    """
    app = FastAPI(
        title="OGX API",
        description="A comprehensive API for building and deploying AI applications",
        version="1.0.0",
        servers=[
            {"url": "http://any-hosted-ogx.com"},
        ],
    )

    # Include routers for all APIs
    protocols = api_protocol_map()
    for api in protocols.keys():
        router = build_fastapi_router(api, None)
        if router is not None:
            app.include_router(router)

    return app
