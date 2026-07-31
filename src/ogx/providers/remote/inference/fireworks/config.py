# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

from pydantic import Field, HttpUrl

from ogx.providers.utils.inference.model_registry import RemoteInferenceProviderConfig
from ogx_api import json_schema_type


@json_schema_type
class FireworksImplConfig(RemoteInferenceProviderConfig):
    """Configuration for the Fireworks AI inference provider."""

    base_url: HttpUrl | None = Field(
        default=HttpUrl("https://api.fireworks.ai/inference/v1"),
        description="The URL for the Fireworks server",
    )

    @classmethod
    def sample_run_config(cls, api_key: str = "${env.FIREWORKS_API_KEY:=}", **kwargs) -> dict[str, Any]:
        return {
            "base_url": "https://api.fireworks.ai/inference/v1",
            "api_key": api_key,
        }
