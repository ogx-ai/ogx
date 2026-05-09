# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

from pydantic import BaseModel, Field, SecretStr

from ogx.providers.utils.inference.model_registry import RemoteInferenceProviderConfig
from ogx_api import json_schema_type


class AnthropicProviderDataValidator(BaseModel):
    """Validates provider-specific request data for Anthropic inference."""

    anthropic_api_key: SecretStr | None = Field(
        default=None,
        description="API key for Anthropic models",
    )


@json_schema_type
class AnthropicConfig(RemoteInferenceProviderConfig):
    """Configuration for the Anthropic inference provider."""

    @classmethod
    def sample_run_config(cls, api_key: str = "${env.ANTHROPIC_API_KEY:=}", **kwargs) -> dict[str, Any]:
        return {
            "api_key": api_key,
        }
