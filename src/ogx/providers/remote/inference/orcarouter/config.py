# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

from pydantic import BaseModel, Field, HttpUrl, SecretStr

from ogx.providers.utils.inference.model_registry import RemoteInferenceProviderConfig
from ogx_api import json_schema_type

DEFAULT_BASE_URL = "https://api.orcarouter.ai/v1"


class OrcaRouterProviderDataValidator(BaseModel):
    """Validates provider-specific request data for OrcaRouter inference."""

    orcarouter_api_key: SecretStr | None = Field(
        default=None,
        description="API key for OrcaRouter models",
    )


@json_schema_type
class OrcaRouterImplConfig(RemoteInferenceProviderConfig):
    """Configuration for the OrcaRouter inference provider."""

    base_url: HttpUrl | None = Field(
        default=HttpUrl(DEFAULT_BASE_URL),
        description="Base URL for the OrcaRouter API",
    )

    @classmethod
    def sample_run_config(cls, api_key: str = "${env.ORCAROUTER_API_KEY:=}", **kwargs) -> dict[str, Any]:
        return {
            "base_url": DEFAULT_BASE_URL,
            "api_key": api_key,
        }
