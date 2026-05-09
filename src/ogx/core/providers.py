# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import asyncio
from typing import Any

from pydantic import BaseModel

from ogx.log import get_logger
from ogx_api import (
    HealthResponse,
    HealthStatus,
    InspectProviderRequest,
    ListProvidersResponse,
    ProviderInfo,
    Providers,
)

from .datatypes import StackConfig
from .utils.config import redact_sensitive_fields

logger = get_logger(name=__name__, category="core")


class ProviderImplConfig(BaseModel):
    """Configuration for the Providers API implementation."""

    config: StackConfig


async def get_provider_impl(config: ProviderImplConfig, deps: dict[str, Any]) -> "ProviderImpl":
    """Create and initialize a ProviderImpl instance.

    Args:
        config: ProviderImplConfig containing the stack configuration.
        deps: Dictionary of API dependencies.

    Returns:
        An initialized ProviderImpl instance.
    """
    impl = ProviderImpl(config, deps)
    await impl.initialize()
    return impl


class ProviderImpl(Providers):
    """Implementation of the Providers API for listing and inspecting configured providers."""

    def __init__(self, config: ProviderImplConfig, deps: dict[str, Any]) -> None:
        self.stack_config = config.config
        self.deps = deps

    async def initialize(self) -> None:
        pass

    async def shutdown(self) -> None:
        logger.debug("ProviderImpl.shutdown")
        pass

    async def list_providers(self) -> ListProvidersResponse:
        run_config = self.stack_config
        safe_config = StackConfig(**redact_sensitive_fields(run_config.model_dump()))
        providers_health = await self.get_providers_health()
        ret = []
        for api, providers in safe_config.providers.items():
            for p in providers:
                # Skip providers that are not enabled
                if p.provider_id is None:
                    continue
                ret.append(
                    ProviderInfo(
                        api=api,
                        provider_id=p.provider_id,
                        provider_type=p.provider_type,
                        config=p.config,
                        health=providers_health.get(api, {}).get(
                            p.provider_id,
                            HealthResponse(
                                status=HealthStatus.NOT_IMPLEMENTED, message="Provider does not implement health check"
                            ),
                        ),
                    )
                )

        return ListProvidersResponse(data=ret)

    async def inspect_provider(self, request: InspectProviderRequest) -> ProviderInfo:
        all_providers = await self.list_providers()
        for p in all_providers.data:
            if p.provider_id == request.provider_id:
                return p

        raise ValueError(f"Provider {request.provider_id} not found")

    async def get_providers_health(self) -> dict[str, dict[str, HealthResponse]]:
        """Get health status for all providers.

        Returns:
            Dict[str, Dict[str, HealthResponse]]: A dictionary mapping API names to provider health statuses.
                Each API maps to a dictionary of provider IDs to their health responses.
        """
        providers_health: dict[str, dict[str, HealthResponse]] = {}

        # The timeout has to be long enough to allow all the providers to be checked, especially in
        # the case of the inference router health check since it checks all registered inference
        # providers.
        # The timeout must not be equal to the one set by health method for a given implementation,
        # otherwise we will miss some providers.
        timeout = 3.0

        async def check_provider_health(impl: Any) -> tuple[str, HealthResponse] | None:
            # Skip special implementations (inspect/providers) that don't have provider specs
            if not hasattr(impl, "__provider_spec__"):
                return None
            api_name = impl.__provider_spec__.api.name
            if not hasattr(impl, "health"):
                return (
                    api_name,
                    HealthResponse(
                        status=HealthStatus.NOT_IMPLEMENTED, message="Provider does not implement health check"
                    ),
                )

            try:
                health = await asyncio.wait_for(impl.health(), timeout=timeout)
                return api_name, health
            except TimeoutError:
                return (
                    api_name,
                    HealthResponse(
                        status=HealthStatus.ERROR, message=f"Health check timed out after {timeout} seconds"
                    ),
                )
            except Exception as e:
                return (
                    api_name,
                    HealthResponse(status=HealthStatus.ERROR, message=f"Health check failed: {str(e)}"),
                )

        # Create tasks for all providers
        tasks = [check_provider_health(impl) for impl in self.deps.values()]

        # Wait for all health checks to complete
        results = await asyncio.gather(*tasks)

        # Organize results by API and provider ID
        for result in results:
            if result is None:  # Skip special implementations
                continue
            api_name, health_response = result
            providers_health[api_name] = health_response

        return providers_health
