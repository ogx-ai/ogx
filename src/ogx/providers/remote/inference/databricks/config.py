# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

from pydantic import BaseModel, Field, HttpUrl, SecretStr

from ogx.providers.utils.inference.model_registry import RemoteInferenceProviderConfig
from ogx_api import json_schema_type


class DatabricksProviderDataValidator(BaseModel):
    """Validates provider-specific request data for Databricks inference."""

    databricks_api_token: SecretStr | None = Field(
        default=None,
        description="API token for Databricks models",
    )


@json_schema_type
class DatabricksImplConfig(RemoteInferenceProviderConfig):
    """Configuration for the Databricks inference provider."""

    base_url: HttpUrl | None = Field(
        default=None,
        description="The URL for the Databricks model serving endpoint (should include /serving-endpoints path)",
    )
    auth_credential: SecretStr | None = Field(
        default=None,
        alias="api_token",
        description="The Databricks API token",
    )

    @classmethod
    def sample_run_config(
        cls,
        base_url: str = "${env.DATABRICKS_HOST:=}",
        api_token: str = "${env.DATABRICKS_TOKEN:=}",
        **kwargs: Any,
    ) -> dict[str, Any]:
        return {
            "base_url": base_url,
            "api_token": api_token,
        }
