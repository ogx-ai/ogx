# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, SecretStr, model_validator

from ogx.providers.utils.forward_headers import validate_forward_headers_config
from ogx_api import json_schema_type


class PassthroughProviderDataValidator(BaseModel):
    """Validates provider-specific request data for passthrough safety forwarding."""

    # extra="allow" because forward_headers key names (e.g. "maas_api_token") are
    # deployer-defined at config time — they can't be declared as typed fields.
    # Without it, Pydantic drops them before build_forwarded_headers() can read them.
    model_config = ConfigDict(extra="allow")

    passthrough_api_key: SecretStr | None = Field(
        default=None,
        description="API key for the downstream safety service",
    )


@json_schema_type
class PassthroughSafetyConfig(BaseModel):
    """Configuration for the passthrough safety provider that forwards to a downstream service."""

    model_config = ConfigDict(extra="forbid")
    base_url: HttpUrl = Field(
        description="Base URL of the downstream safety service (e.g. https://safety.example.com/v1)",
    )
    api_key: SecretStr | None = Field(
        default=None,
        description="API key for the downstream safety service. If set, takes precedence over provider data.",
    )
    forward_headers: dict[str, str] | None = Field(
        default=None,
        description=(
            "Mapping of provider data keys to outbound HTTP header names. "
            "Only keys listed here are forwarded from X-OGX-Provider-Data to the downstream service. "
            "Keys with a __ prefix and core security-sensitive headers (for example Host, "
            "Content-Type, Transfer-Encoding, Cookie) are rejected at config parse time. "
            'Example: {"maas_api_token": "Authorization"}'
        ),
    )
    extra_blocked_headers: list[str] = Field(
        default_factory=list,
        description=(
            "Additional outbound header names to block in forward_headers. "
            "Names are matched case-insensitively and added to the core blocked list. "
            "This can tighten policy but cannot unblock core security-sensitive headers."
        ),
    )

    @model_validator(mode="after")
    def validate_forward_headers(self) -> "PassthroughSafetyConfig":
        validate_forward_headers_config(self.forward_headers, self.extra_blocked_headers)
        return self

    @classmethod
    def sample_run_config(
        cls,
        base_url: str = "${env.PASSTHROUGH_SAFETY_URL}",
        api_key: str = "${env.PASSTHROUGH_SAFETY_API_KEY:=}",
        forward_headers: dict[str, str] | None = None,
        extra_blocked_headers: list[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        config: dict[str, Any] = {
            "base_url": base_url,
            "api_key": api_key,
        }
        if forward_headers:
            config["forward_headers"] = forward_headers
        if extra_blocked_headers:
            config["extra_blocked_headers"] = extra_blocked_headers
        return config
