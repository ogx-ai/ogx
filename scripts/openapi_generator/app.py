# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""
FastAPI app creation for OpenAPI generation.
"""

import inspect
from typing import Any

from fastapi import FastAPI

from ogx.core.resolver import api_protocol_map
from ogx.core.server.fastapi_router_registry import build_fastapi_router
from ogx_api import Api

from .state import _protocol_methods_cache


def _get_protocol_method(api: Api, method_name: str) -> Any | None:
    """
    Get a protocol method function by API and method name.
    Uses caching to avoid repeated lookups.

    Args:
        api: The API enum
        method_name: The method name (function name)

    Returns:
        The function object, or None if not found
    """
    global _protocol_methods_cache

    if _protocol_methods_cache is None:
        _protocol_methods_cache = {}
        protocols = api_protocol_map()
        from ogx_api.tools import SpecialToolGroup, ToolRuntime

        toolgroup_protocols = {
            SpecialToolGroup.rag_tool: ToolRuntime,
        }

        for api_key, protocol in protocols.items():
            method_map: dict[str, Any] = {}
            protocol_methods = inspect.getmembers(protocol, predicate=inspect.isfunction)
            for name, method in protocol_methods:
                method_map[name] = method

            # Handle tool_runtime special case
            if api_key == Api.tool_runtime:
                for tool_group, sub_protocol in toolgroup_protocols.items():
                    sub_protocol_methods = inspect.getmembers(sub_protocol, predicate=inspect.isfunction)
                    for name, method in sub_protocol_methods:
                        if hasattr(method, "__webmethod__"):
                            method_map[f"{tool_group.value}.{name}"] = method

            _protocol_methods_cache[api_key] = method_map

    return _protocol_methods_cache.get(api, {}).get(method_name)


def create_ogx_app() -> FastAPI:
    """
    Create a FastAPI app that represents the OGX API.
    This uses both router-based routes (for migrated APIs) and the existing
    route discovery system for legacy webmethod-based routes.
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
