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

DEFAULT_BASE_URL = "https://api.cerebras.ai/v1"


class CerebrasProviderDataValidator(BaseModel):
    """Validates provider-specific request data for Cerebras inference."""

    cerebras_api_key: SecretStr | None = Field(
        default=None,
        description="API key for Cerebras models",
    )


@json_schema_type
class CerebrasImplConfig(RemoteInferenceProviderConfig):
    """Configuration for the Cerebras inference provider."""

    base_url: HttpUrl | None = Field(
        default=HttpUrl(os.environ.get("CEREBRAS_BASE_URL", DEFAULT_BASE_URL)),
        description="Base URL for the Cerebras API",
    )

    @classmethod
    def sample_run_config(cls, api_key: str = "${env.CEREBRAS_API_KEY:=}", **kwargs) -> dict[str, Any]:
        return {
            "base_url": DEFAULT_BASE_URL,
            "api_key": api_key,
        }
