# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from enum import Enum
from pathlib import PurePath
from typing import Any

from pydantic import SecretStr

_YAML_SAFE_SCALARS = (str, int, float, bool, type(None))


def reveal_secret_fields(data: Any) -> Any:
    """Recursively make model_dump() output safe for yaml.safe_load() roundtrips.

    Converts SecretStr to plaintext, Enum to its value, and Path-like
    objects to strings so the result contains only basic Python types.
    """
    if isinstance(data, dict):
        return {k: reveal_secret_fields(v) for k, v in data.items()}
    if isinstance(data, list):
        return [reveal_secret_fields(v) for v in data]
    if isinstance(data, SecretStr):
        return data.get_secret_value()
    if isinstance(data, Enum):
        return data.value
    if isinstance(data, _YAML_SAFE_SCALARS):
        return data
    if isinstance(data, PurePath):
        return str(data)
    return str(data)


def redact_sensitive_fields(data: dict[str, Any]) -> dict[str, Any]:
    """Redact sensitive information from config before printing."""
    sensitive_patterns = [
        "api_key",
        "api-key",
        "apikey",
        "api_token",
        "api-token",
        "authorization",
        "credential",
        "moderation_headers",
        "password",
        "secret",
        "token",
    ]

    # Specific configuration field names that should NOT be redacted despite containing "token"
    safe_token_fields = ["chunk_size_tokens", "max_tokens", "default_chunk_overlap_tokens", "max_document_tokens"]

    def _redact_value(v: Any) -> Any:
        if isinstance(v, dict):
            return _redact_dict(v)
        elif isinstance(v, list):
            return [_redact_value(i) for i in v]
        return v

    def _redact_dict(d: dict[str, Any]) -> dict[str, Any]:
        result = {}
        for k, v in d.items():
            # Don't redact if it's a safe field
            if any(safe_field in k.lower() for safe_field in safe_token_fields):
                result[k] = _redact_value(v)
            elif any(pattern in k.lower() for pattern in sensitive_patterns):
                result[k] = "********"
            else:
                result[k] = _redact_value(v)
        return result

    return _redact_dict(data)
