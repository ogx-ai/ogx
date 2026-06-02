# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any, Literal

from pydantic import Field

from ogx.providers.utils.common.http import BaseToolRuntimeConfig


class NimbleSearchToolConfig(BaseToolRuntimeConfig):
    """Configuration for the Nimble Search tool runtime."""

    api_key: str | None = Field(
        default=None,
        description="The Nimble API key, sent as a Bearer token",
    )
    max_results: int = Field(
        default=3,
        description="The maximum number of results to return",
    )
    search_depth: Literal["lite", "deep"] = Field(
        default="lite",
        description="Content richness: 'lite' returns title, URL, and description; 'deep' returns full page content",
    )
    country: str = Field(
        default="US",
        description="Country code for geo-targeted results (e.g., 'US', 'GB', 'IL')",
    )
    locale: str = Field(
        default="en",
        description="Language/locale code (e.g., 'en', 'fr', 'de')",
    )

    @classmethod
    def sample_run_config(cls, __distro_dir__: str) -> dict[str, Any]:
        return {
            "api_key": "${env.NIMBLE_API_KEY:=}",
            "max_results": 3,
            "search_depth": "lite",
            "country": "US",
            "locale": "en",
        }
