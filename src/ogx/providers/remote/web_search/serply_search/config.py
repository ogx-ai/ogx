# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import warnings
from typing import Any

from pydantic import Field, SecretStr, model_validator

from ogx.providers.utils.common.http import BaseToolRuntimeConfig

# The Google engine behind Serply caps `num` at 10 per page.
MAX_RESULTS_CEILING = 10


class SerplySearchConfig(BaseToolRuntimeConfig):
    """Configuration for the Serply Search web search backend."""

    api_key: SecretStr | None = Field(
        default=None,
        description="The Serply API key, sent in the X-Api-Key header. Can be overridden per-request via the X-OGX-Provider-Data header.",
    )
    max_results: int = Field(
        default=3,
        description="The maximum number of results to return (at most 10)",
    )

    @model_validator(mode="after")
    def clamp_max_results_to_ceiling(self) -> "SerplySearchConfig":
        if self.max_results > MAX_RESULTS_CEILING:
            warnings.warn(
                f"serply-search max_results={self.max_results} exceeds the Google engine "
                f"ceiling of {MAX_RESULTS_CEILING}; using {MAX_RESULTS_CEILING}.",
                stacklevel=2,
            )
            self.max_results = MAX_RESULTS_CEILING
        return self

    @classmethod
    def sample_run_config(cls, __distro_dir__: str) -> dict[str, Any]:
        return {
            "api_key": "${env.SERPLY_API_KEY:=}",
            "max_results": 3,
        }
