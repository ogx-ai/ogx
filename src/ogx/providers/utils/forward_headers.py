# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import re
from collections.abc import Iterable, Mapping
from typing import Any

from pydantic import BaseModel, SecretStr

from ogx.log import get_logger

logger = get_logger(__name__, category="providers::utils")

CORE_BLOCKED_FORWARD_HEADERS = frozenset(
    {
        # hop-by-hop and framing headers
        "host",
        "content-type",
        "content-length",
        "transfer-encoding",
        "connection",
        "upgrade",
        "te",
        "trailer",
        "cookie",
        "set-cookie",
        # proxy/trust headers — many downstreams use these for source identity and routing
        "forwarded",
        "proxy-authorization",
        "x-forwarded-for",
        "x-forwarded-host",
        "x-forwarded-proto",
        "x-forwarded-prefix",
        "x-real-ip",
    }
)


def normalize_header_name(header_name: str) -> str:
    """Normalize header names so blocking behavior is case-insensitive."""
    return header_name.strip().lower()


_HEADER_NAME_TOKEN_RE = re.compile(r"^[!#$%&'*+\-.^_`|~0-9A-Za-z]+$")


def is_valid_header_name(header_name: str) -> bool:
    """Return True if header_name is a valid HTTP field-name (RFC 7230 token)."""
    stripped = header_name.strip()
    return bool(stripped) and bool(_HEADER_NAME_TOKEN_RE.fullmatch(stripped))


def get_effective_blocked_forward_headers(extra_blocked_headers: Iterable[str] | None = None) -> frozenset[str]:
    """Return core blocked headers plus operator-defined blocked headers."""
    blocked_headers = set(CORE_BLOCKED_FORWARD_HEADERS)
    if extra_blocked_headers:
        blocked_headers.update(
            normalized for header_name in extra_blocked_headers if (normalized := normalize_header_name(header_name))
        )
    return frozenset(blocked_headers)


def validate_forward_headers_config(
    forward_headers: Mapping[str, str] | None,
    extra_blocked_headers: Iterable[str] | None = None,
) -> None:
    """Validate forward_headers policy with core + operator blocked names.

    This policy helper is shared so passthrough inference and future providers
    (for example OpenAIMixin extra headers) can enforce consistent rules.
    """
    errors: list[str] = []

    for header_name in extra_blocked_headers or []:
        if not normalize_header_name(header_name):
            errors.append("extra_blocked_headers contains an empty header name")

    blocked_headers = get_effective_blocked_forward_headers(extra_blocked_headers)
    for provider_key, header_name in (forward_headers or {}).items():
        if provider_key.startswith("__"):
            errors.append(f"provider key '{provider_key}' uses reserved __ prefix")
        if not is_valid_header_name(header_name):
            errors.append(f"header '{header_name}' is not a valid HTTP header name")
        if normalize_header_name(header_name) in blocked_headers:
            errors.append(f"header '{header_name}' is blocked (security-sensitive)")

    if errors:
        raise ValueError(f"invalid forward_headers: {'; '.join(errors)}")


def build_forwarded_headers(
    provider_data: BaseModel | Mapping[str, Any] | None,
    forward_headers: Mapping[str, str] | None,
) -> dict[str, str]:
    """Extract per-request headers from provider data according to the operator allowlist.

    forward_headers maps provider-data key -> outbound HTTP header name.
    Only keys explicitly listed are forwarded; all others are ignored (default-deny).
    Missing keys are silently skipped and do not raise.
    Values are forwarded verbatim (after control-character stripping), so callers
    must include any required prefix in the payload (for example, "Bearer sk-xxx").
    If multiple keys map to the same header name (case-insensitively), the last one in
    iteration order wins, and the emitted header name casing comes from that last mapping.

    Returns an empty dict when forward_headers is None/empty or provider_data is absent.
    """
    if not forward_headers or provider_data is None:
        return {}

    raw_provider_data: Mapping[str, Any]
    if isinstance(provider_data, BaseModel):
        raw_provider_data = provider_data.model_dump()
    else:
        raw_provider_data = provider_data

    # Coalesce header names case-insensitively to avoid emitting duplicates like
    # {"Authorization": "...", "authorization": "..."} which some clients (e.g. httpx)
    # will send as multiple header fields.
    result_by_normalized_name: dict[str, tuple[str, str]] = {}
    for data_key, header_name in forward_headers.items():
        header_name_stripped = header_name.strip()
        normalized_header_name = normalize_header_name(header_name_stripped)
        value = raw_provider_data.get(data_key)
        # skip missing keys and JSON null; treat both as "not provided"
        if value is None:
            continue
        # unwrap SecretStr so we forward the real value, not masked output
        if isinstance(value, SecretStr):
            value = value.get_secret_value()
        # strip ASCII control chars and Unicode line terminators (U+0085/2028/2029) —
        # some HTTP/1.0 proxies treat those as line breaks, enabling header injection
        sanitized = "".join(
            c for c in str(value) if ord(c) >= 0x20 and c != "\x7f" and c not in ("\x85", "\u2028", "\u2029")
        )
        # drop oversized values — downstreams commonly reject headers > 8 KB (HTTP 431),
        # and truncating an auth token would produce a harder-to-debug failure
        if len(sanitized.encode()) > 8192:
            logger.warning(
                "dropping forwarded header '%s': value exceeds 8 KB limit (%d bytes)",
                header_name_stripped,
                len(sanitized.encode()),
            )
            continue
        result_by_normalized_name[normalized_header_name] = (header_name_stripped, sanitized)
    return dict(result_by_normalized_name.values())
