# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

from pydantic import BaseModel, Field, HttpUrl, SecretStr

from ogx.providers.utils.inference.model_registry import RemoteInferenceProviderConfig
from ogx_api import json_schema_type


class GroqProviderDataValidator(BaseModel):
    """Validates provider-specific request data for Groq inference."""

    groq_api_key: SecretStr | None = Field(
        default=None,
        description="API key for Groq models",
    )


@json_schema_type
class GroqConfig(RemoteInferenceProviderConfig):
    """Configuration for the Groq inference provider."""

    base_url: HttpUrl | None = Field(
        default=HttpUrl("https://api.groq.com/openai/v1"),
        description="The URL for the Groq AI server",
    )

    @classmethod
    def sample_run_config(cls, api_key: str = "${env.GROQ_API_KEY:=}", **kwargs) -> dict[str, Any]:
        return {
            "base_url": "https://api.groq.com/openai/v1",
            "api_key": api_key,
        }
