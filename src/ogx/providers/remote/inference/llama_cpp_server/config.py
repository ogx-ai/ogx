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
class LlamaCppServerConfig(RemoteInferenceProviderConfig):
    """Configuration for the llama.cpp server inference provider."""

    base_url: HttpUrl | None = Field(
        default=HttpUrl("http://localhost:8080/v1"),
        description="The URL for the Llama cpp server",
    )

    @classmethod
    def sample_run_config(cls, **kwargs) -> dict[str, Any]:
        return {
            "base_url": "http://localhost:8080/v1",
        }
