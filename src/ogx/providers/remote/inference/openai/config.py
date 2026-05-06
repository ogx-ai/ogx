# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

from pydantic import BaseModel, Field, HttpUrl, SecretStr

from ogx.providers.utils.inference.model_registry import RemoteInferenceProviderConfig
from ogx_api import json_schema_type


class OpenAIProviderDataValidator(BaseModel):
    """Validates provider-specific request data for OpenAI inference."""

    openai_api_key: SecretStr | None = Field(
        default=None,
        description="API key for OpenAI models",
    )


@json_schema_type
class OpenAIConfig(RemoteInferenceProviderConfig):
    """Configuration for the OpenAI inference provider."""

    base_url: HttpUrl | None = Field(
        default=HttpUrl("https://api.openai.com/v1"),
        description="Base URL for OpenAI API",
    )

    @classmethod
    def sample_run_config(
        cls,
        api_key: str = "${env.OPENAI_API_KEY:=}",
        base_url: str = "${env.OPENAI_BASE_URL:=https://api.openai.com/v1}",
        **kwargs,
    ) -> dict[str, Any]:
        return {
            "api_key": api_key,
            "base_url": base_url,
        }
