# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

from pydantic import BaseModel, Field, HttpUrl, SecretStr

from ogx_api import json_schema_type


class TypeSafeProviderDataValidator(BaseModel):
    """Validates provider-specific request data for TypeSafe."""

    typesafe_api_key: SecretStr | None = Field(default=None, description="API key for TypeSafe")


@json_schema_type
class TypeSafeConfig(BaseModel):
    """Configuration for the TypeSafe (Jev) SystemOne judgment provider."""

    base_url: HttpUrl = Field(
        default=HttpUrl("https://api.typesafe.ai"),
        description="Base URL for the TypeSafe API.",
    )
    api_key: SecretStr | None = Field(
        default=None,
        description="API key for TypeSafe. Can be overridden per-request via the X-OGX-Provider-Data header.",
    )

    @classmethod
    def sample_run_config(cls, api_key: str = "${env.TYPESAFE_API_KEY:=}", **kwargs: Any) -> dict[str, Any]:
        return {
            "base_url": "https://api.typesafe.ai",
            "api_key": api_key,
        }
