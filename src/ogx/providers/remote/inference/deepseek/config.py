# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import os
from typing import Any

from pydantic import BaseModel, Field, HttpUrl, SecretStr

from ogx.providers.utils.inference.model_registry import RemoteInferenceProviderConfig
from ogx_api import json_schema_type

DEFAULT_BASE_URL = "https://api.deepseek.com/v1"
DEFAULT_ANTHROPIC_BASE_URL = "https://api.deepseek.com/anthropic"


class DeepSeekProviderDataValidator(BaseModel):
    """Validates provider-specific request data for DeepSeek inference."""

    deepseek_api_key: SecretStr | None = Field(
        default=None,
        description="API key for DeepSeek models",
    )


@json_schema_type
class DeepSeekImplConfig(RemoteInferenceProviderConfig):
    """Configuration for the DeepSeek inference provider."""

    base_url: HttpUrl | None = Field(
        default=HttpUrl(os.environ.get("DEEPSEEK_BASE_URL", DEFAULT_BASE_URL)),
        description="Base URL for the DeepSeek API",
    )

    anthropic_base_url: HttpUrl = Field(
        default=HttpUrl(os.environ.get("DEEPSEEK_ANTHROPIC_BASE_URL", DEFAULT_ANTHROPIC_BASE_URL)),
        description=(
            "Base URL for DeepSeek's native Anthropic-compatible /v1/messages endpoint. This is a "
            "separate host path from the OpenAI-compatible base_url, not a suffix of it."
        ),
    )

    @classmethod
    def sample_run_config(cls, api_key: str = "${env.DEEPSEEK_API_KEY:=}", **kwargs) -> dict[str, Any]:
        return {
            "base_url": DEFAULT_BASE_URL,
            "api_key": api_key,
        }
