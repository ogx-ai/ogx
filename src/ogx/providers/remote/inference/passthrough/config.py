# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

from pydantic import Field, HttpUrl, model_validator

from ogx.providers.utils.forward_headers import validate_forward_headers_config
from ogx.providers.utils.inference.model_registry import RemoteInferenceProviderConfig
from ogx_api import json_schema_type


@json_schema_type
class PassthroughImplConfig(RemoteInferenceProviderConfig):
    """Configuration for the passthrough inference provider."""

    base_url: HttpUrl | None = Field(
        default=None,
        description="The URL for the passthrough endpoint",
    )
    forward_headers: dict[str, str] | None = Field(
        default=None,
        description=(
            "Mapping of X-OGX-Provider-Data keys to outbound HTTP header names. "
            "Only listed keys are forwarded — all others are ignored (default-deny). "
            "Values are forwarded verbatim; include any required prefix in the client payload "
            "(e.g. 'Bearer sk-xxx' not 'sk-xxx' when targeting Authorization). "
            "Header name values should use canonical HTTP casing (e.g. 'Authorization', 'X-Tenant-ID'). "
            "Keys with a __ prefix and core security-sensitive headers (for example Host, "
            "Content-Type, Transfer-Encoding, Cookie) are rejected at config parse time. "
            "When this field is set and auth comes from forwarded headers rather than a static api_key, "
            "the caller must include the required keys in X-OGX-Provider-Data on every request. "
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
    def validate_forward_headers(self) -> "PassthroughImplConfig":
        validate_forward_headers_config(self.forward_headers, self.extra_blocked_headers)
        return self

    @classmethod
    def sample_run_config(
        cls,
        base_url: HttpUrl | None = "${env.PASSTHROUGH_URL}",
        api_key: str = "${env.PASSTHROUGH_API_KEY:=}",
        forward_headers: dict[str, str] | None = None,
        extra_blocked_headers: list[str] | None = None,
        **kwargs,
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
