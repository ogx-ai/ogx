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


class WatsonXProviderDataValidator(BaseModel):
    """Validates provider-specific request data for IBM WatsonX inference."""

    watsonx_project_id: str | None = Field(
        default=None,
        description="IBM WatsonX project ID",
    )
    watsonx_api_key: SecretStr | None = None


@json_schema_type
class WatsonXConfig(RemoteInferenceProviderConfig):
    """Configuration for the IBM WatsonX inference provider."""

    base_url: HttpUrl | None = Field(
        default_factory=lambda: os.getenv("WATSONX_BASE_URL", "https://us-south.ml.cloud.ibm.com"),
        description="A base url for accessing the watsonx.ai",
    )
    project_id: str | None = Field(
        default=None,
        description="The watsonx.ai project ID",
    )
    timeout: int = Field(
        default=60,
        description="Timeout for the HTTP requests",
    )

    @classmethod
    def sample_run_config(cls, **kwargs) -> dict[str, Any]:
        return {
            "base_url": "${env.WATSONX_BASE_URL:=https://us-south.ml.cloud.ibm.com}",
            "api_key": "${env.WATSONX_API_KEY:=}",
            "project_id": "${env.WATSONX_PROJECT_ID:=}",
        }
