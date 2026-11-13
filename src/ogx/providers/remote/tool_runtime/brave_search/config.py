# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

from pydantic import Field, HttpUrl, SecretStr

from ogx.providers.utils.common.http import BaseToolRuntimeConfig

DEFAULT_BASE_URL = "https://api.search.brave.com"


class BraveSearchToolConfig(BaseToolRuntimeConfig):
    """Configuration for the Brave Search tool runtime."""

    base_url: HttpUrl = Field(
        default=HttpUrl(DEFAULT_BASE_URL),
        description="Base URL of the Brave Search API. Set it to point the tool at a Brave-compatible proxy or alternate backend.",
    )
    api_key: SecretStr | None = Field(
        default=None,
        description="The Brave Search API Key. Can be overridden per-request via X-OGX-Provider-Data header.",
    )
    max_results: int = Field(
        default=3,
        description="The maximum number of results to return",
    )

    @classmethod
    def sample_run_config(cls, __distro_dir__: str) -> dict[str, Any]:
        return {
            "base_url": "${env.BRAVE_SEARCH_BASE_URL:=" + DEFAULT_BASE_URL + "}",
            "api_key": "${env.BRAVE_SEARCH_API_KEY:=}",
            "max_results": 3,
        }
