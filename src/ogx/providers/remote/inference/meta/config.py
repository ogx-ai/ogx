# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

from pydantic import BaseModel, Field, HttpUrl, SecretStr

from ogx.providers.utils.inference.model_registry import RemoteInferenceProviderConfig
from ogx_api import json_schema_type

DEFAULT_BASE_URL = "https://api.meta.ai/v1"


class MetaProviderDataValidator(BaseModel):
    """Validates provider-specific request data for Meta AI inference."""

    meta_api_key: SecretStr | None = Field(
        default=None,
        description="API key for Meta AI models",
    )


@json_schema_type
class MetaConfig(RemoteInferenceProviderConfig):
    """Configuration for the Meta AI inference provider."""

    base_url: HttpUrl | None = Field(
        default=HttpUrl(DEFAULT_BASE_URL),
        description="The URL for the Meta AI API server",
    )

    @classmethod
    def sample_run_config(cls, api_key: str = "${env.META_API_KEY:=}", **kwargs) -> dict[str, Any]:
        return {
            "base_url": DEFAULT_BASE_URL,
            "api_key": api_key,
        }
