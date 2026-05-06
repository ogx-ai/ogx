# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, SecretStr

from ogx.providers.utils.inference.model_registry import RemoteInferenceProviderConfig
from ogx_api import json_schema_type


class VertexAIProviderDataValidator(BaseModel):
    """Validates provider-specific request data for Google Vertex AI inference."""

    vertex_project: str | None = Field(
        default=None,
        description="Google Cloud project ID for Vertex AI",
    )
    vertex_location: str | None = Field(
        default=None,
        description="Google Cloud location for Vertex AI (e.g., global)",
    )
    vertex_access_token: SecretStr | None = None


@json_schema_type
class VertexAIConfig(RemoteInferenceProviderConfig):
    """Configuration for the Vertex AI inference provider.

    Supports either Application Default Credentials (default behavior) or an
    explicit OAuth access token provided via `access_token` (alias for
    `auth_credential`) for short-lived credential injection.
    """

    model_config = ConfigDict(populate_by_name=True)

    auth_credential: SecretStr | None = Field(
        default=None,
        alias="access_token",
        exclude=True,
        description="Optional access token for Vertex AI authentication. When set, this is used instead of Application Default Credentials (ADC).",
        json_schema_extra={"env": "VERTEX_AI_ACCESS_TOKEN"},
    )

    project: str = Field(
        description="Google Cloud project ID for Vertex AI",
    )
    location: str = Field(
        default="global",
        description="Google Cloud location for Vertex AI",
    )

    @classmethod
    def sample_run_config(
        cls,
        project: str = "${env.VERTEX_AI_PROJECT:=}",
        location: str = "${env.VERTEX_AI_LOCATION:=global}",
        **kwargs,
    ) -> dict[str, Any]:
        """Return sample run config."""
        return {
            "project": project,
            "location": location,
        }
