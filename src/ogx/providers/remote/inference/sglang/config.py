# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from pydantic import Field, HttpUrl, SecretStr

from ogx.providers.utils.inference.model_registry import (
    RemoteInferenceProviderConfig,
)
from ogx_api import json_schema_type


@json_schema_type
class SGLangInferenceAdapterConfig(RemoteInferenceProviderConfig):
    """Configuration for the remote SGLang inference provider."""

    base_url: HttpUrl | None = Field(
        default=None,
        description="The URL for the SGLang server, e.g. http://localhost:30000/v1",
    )
    max_tokens: int = Field(
        default=4096,
        description="Maximum number of tokens to generate.",
    )
    auth_credential: SecretStr | None = Field(
        default=None,
        alias="api_token",
        description="The API token (the server's --api-key), if any",
    )

    @classmethod
    def sample_run_config(
        cls,
        base_url: str = "${env.SGLANG_URL:=}",
        **kwargs,
    ):
        return {
            "base_url": base_url,
            "max_tokens": "${env.SGLANG_MAX_TOKENS:=4096}",
            "api_token": "${env.SGLANG_API_TOKEN:=fake}",
            "network": {
                "tls": {
                    "verify": "${env.SGLANG_TLS_VERIFY:=true}",
                },
            },
        }
