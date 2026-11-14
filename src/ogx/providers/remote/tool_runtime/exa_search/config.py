# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

from pydantic import Field, SecretStr

from ogx.providers.utils.common.http import BaseToolRuntimeConfig


class ExaSearchToolConfig(BaseToolRuntimeConfig):
    """Configuration for the Exa Search tool runtime."""

    api_key: SecretStr | None = Field(
        default=None,
        description="The Exa API Key. Can be overridden per-request via X-OGX-Provider-Data header.",
    )
    max_results: int = Field(
        default=3,
        description="The maximum number of results to return",
    )

    @classmethod
    def sample_run_config(cls, __distro_dir__: str) -> dict[str, Any]:
        return {
            "api_key": "${env.EXA_API_KEY:=}",
            "max_results": 3,
        }
