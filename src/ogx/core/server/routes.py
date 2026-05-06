# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ogx.core.server.fastapi_router_registry import (
    _ROUTER_FACTORIES,
    build_fastapi_router,
    get_router_routes,
)
from ogx_api import Api
from ogx_api.router_utils import PUBLIC_ROUTE_KEY


@dataclass
class RouteAuthInfo:
    """Authentication metadata for a route."""

    require_authentication: bool = True


EndpointFunc = Callable[..., Any]
PathParams = dict[str, str]
RouteInfo = tuple[EndpointFunc, str, RouteAuthInfo]
PathImpl = dict[str, RouteInfo]
RouteImpls = dict[str, PathImpl]
RouteMatch = tuple[EndpointFunc, PathParams, str, RouteAuthInfo]


def _convert_path_to_regex(path: str) -> str:
    # Convert {param} to named capture groups
    # handle {param:path} as well which allows for forward slashes in the param value
    pattern = re.sub(
        r"{(\w+)(?::path)?}",
        lambda m: f"(?P<{m.group(1)}>{'[^/]+' if not m.group(0).endswith(':path') else '.+'})",
        path,
    )

    return f"^{pattern}$"


def initialize_route_impls(impls: dict[Api, Any]) -> RouteImpls:
    route_impls: RouteImpls = {}

    for api_name in _ROUTER_FACTORIES.keys():
        api = Api(api_name)
        if api not in impls:
            continue
        impl = impls[api]
        router = build_fastapi_router(api, impl)
        if router:
            router_routes = get_router_routes(router)
            for route in router_routes:
                func = route.endpoint
                if func is None:
                    continue

                available_methods = [m for m in (route.methods or []) if m != "HEAD"]
                if not available_methods:
                    continue
                method = available_methods[0].lower()

                if method not in route_impls:
                    route_impls[method] = {}

                # Routes with openapi_extra[PUBLIC_ROUTE_KEY]=True don't require authentication
                is_public = (route.openapi_extra or {}).get(PUBLIC_ROUTE_KEY, False)
                auth_info = RouteAuthInfo(
                    require_authentication=not is_public,
                )
                route_impls[method][_convert_path_to_regex(route.path)] = (
                    func,
                    route.path,
                    auth_info,
                )

    return route_impls


def find_matching_route(method: str, path: str, route_impls: RouteImpls) -> RouteMatch:
    """Find the matching endpoint implementation for a given method and path.

    Returns a tuple of (endpoint_function, path_params, route_path, webmethod_metadata).

    Raises ValueError if no matching endpoint is found.
    """
    impls = route_impls.get(method.lower())
    if not impls:
        raise ValueError(f"No endpoint found for {path}")

    for regex, (func, route_path, webmethod) in impls.items():
        match = re.match(regex, path)
        if match:
            path_params = match.groupdict()
            return func, path_params, route_path, webmethod

    raise ValueError(f"No endpoint found for {path}")
