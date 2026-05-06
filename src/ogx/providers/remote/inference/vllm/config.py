# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import warnings
from pathlib import Path

from pydantic import Field, HttpUrl, SecretStr, model_validator

from ogx.providers.utils.inference.model_registry import (
    NetworkConfig,
    RemoteInferenceProviderConfig,
    TLSConfig,
)
from ogx_api import json_schema_type


@json_schema_type
class VLLMInferenceAdapterConfig(RemoteInferenceProviderConfig):
    """Configuration for the remote vLLM inference provider."""

    base_url: HttpUrl | None = Field(
        default=None,
        description="The URL for the vLLM model serving endpoint",
    )
    max_tokens: int = Field(
        default=4096,
        description="Maximum number of tokens to generate.",
    )
    auth_credential: SecretStr | None = Field(
        default=None,
        alias="api_token",
        description="The API token",
    )
    tls_verify: bool | str | None = Field(
        default=None,
        deprecated=True,
        description="DEPRECATED: Use 'network.tls.verify' instead. Whether to verify TLS certificates. "
        "Can be a boolean or a path to a CA certificate file.",
    )

    @model_validator(mode="after")
    def migrate_tls_verify_to_network(self) -> "VLLMInferenceAdapterConfig":
        """Migrate legacy tls_verify to network.tls.verify for backward compatibility."""
        if self.tls_verify is not None:
            warnings.warn(
                "The 'tls_verify' config option is deprecated. Please use 'network.tls.verify' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            # Convert string path to Path if needed
            if isinstance(self.tls_verify, str):
                verify_value: bool | Path = Path(self.tls_verify)
            else:
                verify_value = self.tls_verify

            if self.network is None:
                self.network = NetworkConfig(tls=TLSConfig(verify=verify_value))
            elif self.network.tls is None:
                self.network.tls = TLSConfig(verify=verify_value)
        return self

    @classmethod
    def sample_run_config(
        cls,
        base_url: str = "${env.VLLM_URL:=}",
        **kwargs,
    ):
        return {
            "base_url": base_url,
            "max_tokens": "${env.VLLM_MAX_TOKENS:=4096}",
            "api_token": "${env.VLLM_API_TOKEN:=fake}",
            "network": {
                "tls": {
                    "verify": "${env.VLLM_TLS_VERIFY:=true}",
                },
            },
        }
