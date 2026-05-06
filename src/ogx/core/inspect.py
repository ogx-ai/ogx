# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from importlib.metadata import version
from typing import Any

from pydantic import BaseModel

from ogx.core.datatypes import StackConfig
from ogx.core.server.fastapi_router_registry import (
    _ROUTER_FACTORIES,
    build_fastapi_router,
    get_router_routes,
)
from ogx_api import (
    Api,
    HealthInfo,
    HealthStatus,
    Inspect,
    ListRoutesResponse,
    RouteInfo,
    VersionInfo,
)


class DistributionInspectConfig(BaseModel):
    """Configuration for the Inspect API implementation."""

    config: StackConfig


async def get_provider_impl(config: DistributionInspectConfig, deps: dict[str, Any]) -> "DistributionInspectImpl":
    """Create and initialize a DistributionInspectImpl instance.

    Args:
        config: DistributionInspectConfig containing the stack configuration.
        deps: Dictionary of API dependencies.

    Returns:
        An initialized DistributionInspectImpl instance.
    """
    impl = DistributionInspectImpl(config, deps)
    await impl.initialize()
    return impl


class DistributionInspectImpl(Inspect):
    """Implementation of the Inspect API providing route listing, health, and version endpoints."""

    def __init__(self, config: DistributionInspectConfig, deps: dict[str, Any]) -> None:
        self.stack_config = config.config
        self.deps = deps

    async def initialize(self) -> None:
        pass

    async def list_routes(self, api_filter: str | None = None) -> ListRoutesResponse:
        config: StackConfig = self.stack_config

        # Helper function to get provider types for an API
        def _get_provider_types(api: Api) -> list[str]:
            if api.value in ["providers", "inspect"]:
                return []  # These APIs don't have "real" providers — they're internal to the stack
            providers = config.providers.get(api.value, [])
            return [p.provider_type for p in providers] if providers else []

        # Helper function to determine if a router route should be included based on api_filter
        def _should_include_router_route(route: Any, router_prefix: str | None) -> bool:
            """Check if a router-based route should be included based on api_filter."""
            # Check deprecated status
            route_deprecated = getattr(route, "deprecated", False) or False

            if api_filter is None:
                return not route_deprecated
            elif api_filter == "deprecated":
                return route_deprecated
            else:
                if router_prefix:
                    prefix_level = router_prefix.lstrip("/")
                    return not route_deprecated and prefix_level == api_filter
                return not route_deprecated

        ret = []
        for api_name in _ROUTER_FACTORIES.keys():
            api = Api(api_name)
            router = build_fastapi_router(api, None)
            if router:
                for route in get_router_routes(router):
                    if _should_include_router_route(route, router.prefix):
                        if route.methods is not None:
                            available_methods = [m for m in route.methods if m != "HEAD"]
                            if available_methods:
                                ret.append(
                                    RouteInfo(
                                        route=route.path,
                                        method=available_methods[0],
                                        provider_types=_get_provider_types(api),
                                    )
                                )

        return ListRoutesResponse(data=ret)

    async def health(self) -> HealthInfo:
        return HealthInfo(status=HealthStatus.OK)

    async def version(self) -> VersionInfo:
        return VersionInfo(version=version("ogx"))

    async def shutdown(self) -> None:
        pass
