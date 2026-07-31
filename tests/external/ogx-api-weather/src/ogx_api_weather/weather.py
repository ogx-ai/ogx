# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Protocol

from fastapi import APIRouter

from ogx_api import Api, ProviderSpec, RemoteProviderSpec


def available_providers() -> list[ProviderSpec]:
    return [
        RemoteProviderSpec(
            api=Api.weather,
            provider_type="remote::kaze",
            config_class="ogx_provider_kaze.KazeProviderConfig",
            adapter_type="kaze",
            module="ogx_provider_kaze",
            pip_packages=["ogx_provider_kaze"],
        ),
    ]


class WeatherProvider(Protocol):
    """
    A protocol for the Weather API.
    """

    async def get_available_locations() -> dict[str, list[str]]:
        """
        Get the available locations.
        """
        ...


def create_router(impl: WeatherProvider) -> APIRouter:
    router = APIRouter()

    @router.get("/v1/weather/locations")
    async def get_available_locations() -> dict[str, list[str]]:
        return await impl.get_available_locations()

    return router
