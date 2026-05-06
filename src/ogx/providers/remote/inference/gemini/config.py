# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

from pydantic import BaseModel, Field, SecretStr, model_validator

from ogx.providers.utils.inference.model_registry import RemoteInferenceProviderConfig
from ogx_api import json_schema_type


class GeminiProviderDataValidator(BaseModel):
    """Validates provider-specific request data for Google Gemini inference."""

    gemini_api_key: SecretStr | None = Field(
        default=None,
        description="API key for Gemini models",
    )


@json_schema_type
class GeminiConfig(RemoteInferenceProviderConfig):
    """Configuration for the Google Gemini inference provider.

    Supports either a static API key (``api_key`` / ``GEMINI_API_KEY``) or an
    OAuth2 access token (``access_token`` / ``GEMINI_ACCESS_TOKEN``) for
    short-lived credential injection (e.g. ``gcloud auth application-default
    print-access-token``).  When both are set, ``access_token`` takes precedence.
    """

    access_token: SecretStr | None = Field(
        default=None,
        description="OAuth2 access token for Gemini. When set, used instead of api_key for Bearer authentication.",
    )
    project: str | None = Field(
        default=None,
        description="Google Cloud project ID for quota attribution when using OAuth/ADC credentials.",
    )

    @model_validator(mode="after")
    def _validate_auth(self) -> "GeminiConfig":
        has_api_key = self.auth_credential and self.auth_credential.get_secret_value()
        has_access_token = self.access_token and self.access_token.get_secret_value()

        if has_api_key and has_access_token:
            raise ValueError("api_key and access_token are mutually exclusive — set one or the other, not both")
        if has_access_token and not self.project:
            raise ValueError(
                "project is required when using access_token (Google APIs need a quota project for OAuth/ADC credentials)"
            )
        if self.project and not has_access_token:
            raise ValueError("project requires access_token — api_key authentication does not use a quota project")
        return self

    @classmethod
    def sample_run_config(
        cls,
        api_key: str = "${env.GEMINI_API_KEY:=}",
        access_token: str = "${env.GEMINI_ACCESS_TOKEN:=}",
        project: str = "${env.GEMINI_AI_PROJECT:=}",
        **kwargs,
    ) -> dict[str, Any]:
        return {
            "api_key": api_key,
            "access_token": access_token,
            "project": project,
        }
