# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

from pydantic import BaseModel, Field, SecretStr

from ogx_api import json_schema_type


class SambaNovaProviderDataValidator(BaseModel):
    """Validates provider-specific request data for SambaNova safety."""

    sambanova_api_key: SecretStr | None = Field(
        default=None,
        description="Sambanova Cloud API key",
    )


@json_schema_type
class SambaNovaSafetyConfig(BaseModel):
    """Configuration for the SambaNova safety provider."""

    url: str = Field(
        default="https://api.sambanova.ai/v1",
        description="The URL for the SambaNova AI server",
    )
    api_key: SecretStr | None = Field(
        default=None,
        description="The SambaNova cloud API Key",
    )

    @classmethod
    def sample_run_config(cls, api_key: str = "${env.SAMBANOVA_API_KEY:=}", **kwargs) -> dict[str, Any]:
        return {
            "url": "https://api.sambanova.ai/v1",
            "api_key": api_key,
        }
