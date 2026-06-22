# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

from pydantic import Field, SecretStr

from ogx.providers.utils.common.http import BaseToolRuntimeConfig


class BingSearchToolConfig(BaseToolRuntimeConfig):
    """Configuration for Bing Search Tool Runtime"""

    api_key: SecretStr | None = Field(
        default=None,
        description="The Bing Search API Key. Can be overridden per-request via X-OGX-Provider-Data header.",
    )
    top_k: int = 3

    @classmethod
    def sample_run_config(cls, __distro_dir__: str, **kwargs: Any) -> dict[str, Any]:
        return {
            "api_key": "${env.BING_API_KEY:=}",
        }
