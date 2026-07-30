# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

from pydantic import BaseModel, Field, HttpUrl, SecretStr

from ogx.providers.utils.inference.model_registry import RemoteInferenceProviderConfig
from ogx_api import json_schema_type


class SambaNovaProviderDataValidator(BaseModel):
    """Validates provider-specific request data for SambaNova inference."""

    sambanova_api_key: SecretStr | None = Field(
        default=None,
        description="Sambanova Cloud API key",
    )


@json_schema_type
class SambaNovaImplConfig(RemoteInferenceProviderConfig):
    """Configuration for the SambaNova inference provider."""

    base_url: HttpUrl | None = Field(
        default=HttpUrl("https://api.sambanova.ai/v1"),
        description="The URL for the SambaNova AI server",
    )

    @classmethod
    def sample_run_config(cls, api_key: str = "${env.SAMBANOVA_API_KEY:=}", **kwargs) -> dict[str, Any]:
        return {
            "base_url": "https://api.sambanova.ai/v1",
            "api_key": api_key,
        }
