# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from ogx_api_weather.weather import WeatherProvider

from .config import KazeProviderConfig


class WeatherKazeAdapter(WeatherProvider):
    """Kaze weather provider implementation."""

    def __init__(
        self,
        config: KazeProviderConfig,
    ) -> None:
        self.config = config

    async def initialize(self) -> None:
        pass

    async def get_available_locations(self) -> dict[str, list[str]]:
        """Get available weather locations."""
        return {"locations": ["Paris", "Tokyo"]}
