# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import asyncio
from unittest.mock import MagicMock, patch

import pytest
from pydantic import SecretStr

from ogx.core.request_headers import RequestProviderDataContext
from ogx.providers.remote.inference.passthrough import PassthroughProviderDataValidator
from ogx.providers.remote.inference.passthrough.config import PassthroughImplConfig
from ogx.providers.remote.inference.passthrough.passthrough import PassthroughInferenceAdapter
from ogx.providers.utils.forward_headers import (
    build_forwarded_headers,
    get_effective_blocked_forward_headers,
    validate_forward_headers_config,
)

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

_PROVIDER_VALIDATOR_PATH = "ogx.providers.remote.inference.passthrough.PassthroughProviderDataValidator"


def _make_adapter(
    forward_headers: dict[str, str] | None = None,
    base_url: str = "http://downstream.example.com",
    api_key: str = "sk-test",
) -> PassthroughInferenceAdapter:
    config = PassthroughImplConfig(
        base_url=base_url,  # type: ignore[arg-type]
        api_key=SecretStr(api_key),
        forward_headers=forward_headers,
    )
    adapter = PassthroughInferenceAdapter(config)
    spec = MagicMock()
    spec.provider_data_validator = _PROVIDER_VALIDATOR_PATH
    object.__setattr__(adapter, "__provider_spec__", spec)
    return adapter


# ---------------------------------------------------------------------------
# build_forwarded_headers utility
# ---------------------------------------------------------------------------


class TestBuildForwardedHeaders:
    def test_returns_empty_dict_when_forward_headers_none(self):
        assert build_forwarded_headers({"token": "Bearer abc"}, None) == {}

    def test_returns_empty_dict_when_forward_headers_empty(self):
        # explicit empty dict should behave the same as None (section 8c/9b three-state check)
        assert build_forwarded_headers({"token": "Bearer abc"}, {}) == {}

    def test_returns_empty_dict_when_no_provider_data(self):
        assert build_forwarded_headers(None, {"token": "Authorization"}) == {}

    def test_maps_listed_key_to_header_name(self):
        result = build_forwarded_headers(
            {"maas_api_token": "Bearer sk-xyz"},
            {"maas_api_token": "Authorization"},
        )
        assert result == {"Authorization": "Bearer sk-xyz"}

    def test_silently_skips_missing_keys(self):
        result = build_forwarded_headers(
            {"other_key": "value"},
            {"missing_key": "X-Custom-Auth"},
        )
        assert result == {}

    def test_default_deny_unlisted_keys_not_forwarded(self):
        result = build_forwarded_headers(
            {"token": "Bearer abc", "secret": "should-not-forward"},
            {"token": "Authorization"},
        )
        assert "secret" not in result
        assert result == {"Authorization": "Bearer abc"}

    def test_multiple_keys_forwarded_together(self):
        result = build_forwarded_headers(
            {"tok": "Bearer abc", "tid": "acme"},
            {"tok": "Authorization", "tid": "X-Tenant-ID"},
        )
        assert result == {"Authorization": "Bearer abc", "X-Tenant-ID": "acme"}

    def test_strips_crlf_from_values(self):
        # CRLF in a header value is a header injection vector
        result = build_forwarded_headers(
            {"token": "Bearer real\r\nX-Injected: evil"},
            {"token": "Authorization"},
        )
        assert result == {"Authorization": "Bearer realX-Injected: evil"}
        assert "\r" not in result["Authorization"]
        assert "\n" not in result["Authorization"]

    def test_strips_all_control_characters(self):
        # null byte, tab, and other control chars stripped
        result = build_forwarded_headers(
            {"token": "Bearer \x00abc\tdef"},
            {"token": "Authorization"},
        )
        assert result == {"Authorization": "Bearer abcdef"}

    def test_null_json_value_is_skipped(self):
        # JSON null treated same as a missing key — not forwarded
        result = build_forwarded_headers(
            {"flag": None},
            {"flag": "X-Flag"},
        )
        assert result == {}

    def test_unwraps_secretstr_values(self):
        result = build_forwarded_headers(
            {"token": SecretStr("Bearer sk-secret")},
            {"token": "Authorization"},
        )
        assert result == {"Authorization": "Bearer sk-secret"}

    def test_coalesces_header_names_case_insensitive(self):
        result = build_forwarded_headers(
            {"a": "Bearer A", "b": "Bearer B"},
            {"a": "Authorization", "b": "authorization"},
        )
        auth_keys = [k for k in result.keys() if k.lower() == "authorization"]
        assert len(auth_keys) == 1
        assert result[auth_keys[0]] == "Bearer B"

    def test_strips_whitespace_from_header_names(self):
        result = build_forwarded_headers(
            {"token": "Bearer abc"},
            {"token": "  Authorization  "},
        )
        assert result == {"Authorization": "Bearer abc"}


# ---------------------------------------------------------------------------
# shared forward-header policy helpers
# ---------------------------------------------------------------------------


class TestForwardHeaderPolicyHelpers:
    def test_effective_blocked_headers_include_core_defaults(self):
        blocked = get_effective_blocked_forward_headers()
        assert "host" in blocked
        assert "content-type" in blocked

    def test_effective_blocked_headers_include_operator_overrides_case_insensitive(self):
        blocked = get_effective_blocked_forward_headers(["X-Internal-Debug", "x-shadow-auth"])
        assert "x-internal-debug" in blocked
        assert "x-shadow-auth" in blocked

    def test_validate_forward_headers_rejects_empty_extra_blocked_name(self):
        with pytest.raises(ValueError, match="extra_blocked_headers contains an empty header name"):
            validate_forward_headers_config({"tenant_id": "X-Tenant-ID"}, ["   "])

    def test_validate_forward_headers_rejects_invalid_header_names(self):
        with pytest.raises(ValueError, match="not a valid HTTP header name"):
            validate_forward_headers_config({"tenant_id": "X Tenant ID"})


# ---------------------------------------------------------------------------
# _get_openai_client — integration with the adapter
# ---------------------------------------------------------------------------


class TestPassthroughForwardHeaders:
    @patch("ogx.providers.remote.inference.passthrough.passthrough.AsyncOpenAI")
    def test_forwards_listed_key_as_correct_header_name(self, mock_openai: MagicMock):
        adapter = _make_adapter(forward_headers={"maas_api_token": "Authorization"})
        with RequestProviderDataContext({"maas_api_token": "Bearer sk-abc123", "other": "ignored"}):
            adapter._get_openai_client()

        call_kwargs = mock_openai.call_args[1]
        # forwarded Authorization is in default_headers; static api_key also sets it
        # via _build_request_headers — static key wins, so check default_headers has it
        assert "Authorization" in (call_kwargs.get("default_headers") or {})

    @patch("ogx.providers.remote.inference.passthrough.passthrough.AsyncOpenAI")
    def test_default_deny_unlisted_keys_not_forwarded(self, mock_openai: MagicMock):
        adapter = _make_adapter(forward_headers={"token": "Authorization"})
        with RequestProviderDataContext({"token": "Bearer abc", "secret": "must-not-leak"}):
            adapter._get_openai_client()

        call_kwargs = mock_openai.call_args[1]
        headers = call_kwargs.get("default_headers") or {}
        assert "secret" not in str(headers)
        assert "must-not-leak" not in str(headers)

    @patch("ogx.providers.remote.inference.passthrough.passthrough.AsyncOpenAI")
    def test_static_api_key_used_when_no_forward_headers(self, mock_openai: MagicMock):
        adapter = _make_adapter(forward_headers=None)
        with RequestProviderDataContext({"maas_api_token": "Bearer abc"}):
            adapter._get_openai_client()

        call_kwargs = mock_openai.call_args[1]
        headers = call_kwargs.get("default_headers") or {}
        assert headers == {"Authorization": "Bearer sk-test"}

    @patch("ogx.providers.remote.inference.passthrough.passthrough.AsyncOpenAI")
    def test_missing_key_in_provider_data_skips_silently(self, mock_openai: MagicMock):
        adapter = _make_adapter(forward_headers={"missing_key": "X-Custom-Auth"})
        with RequestProviderDataContext({"other_key": "irrelevant"}):
            adapter._get_openai_client()

        call_kwargs = mock_openai.call_args[1]
        headers = call_kwargs.get("default_headers") or {}
        assert "X-Custom-Auth" not in headers

    @patch("ogx.providers.remote.inference.passthrough.passthrough.AsyncOpenAI")
    def test_provider_data_passthrough_api_key_used_when_no_static_key(self, mock_openai: MagicMock):
        config = PassthroughImplConfig(
            base_url="http://downstream.example.com",  # type: ignore[arg-type]
            forward_headers={"tenant_id": "X-Tenant-ID"},
        )
        adapter = PassthroughInferenceAdapter(config)
        spec = MagicMock()
        spec.provider_data_validator = _PROVIDER_VALIDATOR_PATH
        object.__setattr__(adapter, "__provider_spec__", spec)

        with RequestProviderDataContext({"passthrough_api_key": "sk-user-key", "tenant_id": "acme"}):
            adapter._get_openai_client()

        call_kwargs = mock_openai.call_args[1]
        headers = call_kwargs.get("default_headers") or {}
        assert headers.get("Authorization") == "Bearer sk-user-key"
        assert headers.get("X-Tenant-ID") == "acme"

    @patch("ogx.providers.remote.inference.passthrough.passthrough.AsyncOpenAI")
    def test_empty_static_api_key_falls_back_to_provider_data_key(self, mock_openai: MagicMock):
        config = PassthroughImplConfig(
            base_url="http://downstream.example.com",  # type: ignore[arg-type]
            api_key=SecretStr(""),
            forward_headers={"tenant_id": "X-Tenant-ID"},
        )
        adapter = PassthroughInferenceAdapter(config)
        spec = MagicMock()
        spec.provider_data_validator = _PROVIDER_VALIDATOR_PATH
        object.__setattr__(adapter, "__provider_spec__", spec)

        with RequestProviderDataContext({"passthrough_api_key": "sk-user-key", "tenant_id": "acme"}):
            adapter._get_openai_client()

        call_kwargs = mock_openai.call_args[1]
        headers = call_kwargs.get("default_headers") or {}
        assert headers.get("Authorization") == "Bearer sk-user-key"
        assert headers.get("X-Tenant-ID") == "acme"

    @patch("ogx.providers.remote.inference.passthrough.passthrough.AsyncOpenAI")
    def test_forwarded_auth_used_when_no_static_credential(self, mock_openai: MagicMock):
        # forward_headers supplies auth; no static api_key → api_key="" so SDK adds no extra Authorization
        config = PassthroughImplConfig(
            base_url="http://downstream.example.com",  # type: ignore[arg-type]
            forward_headers={"token": "Authorization"},
        )
        adapter = PassthroughInferenceAdapter(config)
        spec = MagicMock()
        spec.provider_data_validator = _PROVIDER_VALIDATOR_PATH
        object.__setattr__(adapter, "__provider_spec__", spec)

        with RequestProviderDataContext({"token": "Bearer real-token"}):
            adapter._get_openai_client()

        call_kwargs = mock_openai.call_args[1]
        assert call_kwargs["api_key"] == ""
        assert call_kwargs["default_headers"] == {"Authorization": "Bearer real-token"}

    @patch("ogx.providers.remote.inference.passthrough.passthrough.AsyncOpenAI")
    def test_tenant_only_forward_adds_no_authorization(self, mock_openai: MagicMock):
        # forward_headers → X-Tenant-ID only, no api_key → Authorization must not appear
        config = PassthroughImplConfig(
            base_url="http://downstream.example.com",  # type: ignore[arg-type]
            forward_headers={"tenant_id": "X-Tenant-ID"},
        )
        adapter = PassthroughInferenceAdapter(config)
        spec = MagicMock()
        spec.provider_data_validator = _PROVIDER_VALIDATOR_PATH
        object.__setattr__(adapter, "__provider_spec__", spec)

        with RequestProviderDataContext({"tenant_id": "acme"}):
            adapter._get_openai_client()

        call_kwargs = mock_openai.call_args[1]
        assert call_kwargs["api_key"] == ""
        headers = call_kwargs.get("default_headers") or {}
        assert "Authorization" not in headers
        assert headers.get("X-Tenant-ID") == "acme"

    @patch("ogx.providers.remote.inference.passthrough.passthrough.AsyncOpenAI")
    def test_static_key_takes_priority_over_forwarded_auth(self, mock_openai: MagicMock):
        # static api_key + forwarded auth token → static wins
        adapter = _make_adapter(
            forward_headers={"user_token": "Authorization"},
            api_key="sk-static",
        )
        with RequestProviderDataContext({"user_token": "Bearer user-token"}):
            adapter._get_openai_client()

        call_kwargs = mock_openai.call_args[1]
        headers = call_kwargs.get("default_headers") or {}
        assert headers.get("Authorization") == "Bearer sk-static"


# ---------------------------------------------------------------------------
# _build_request_headers — unit tests for the header assembly logic
# ---------------------------------------------------------------------------


class TestBuildRequestHeaders:
    def _make_no_static_key_adapter(self, forward_headers: dict[str, str] | None = None) -> PassthroughInferenceAdapter:
        config = PassthroughImplConfig(
            base_url="http://downstream.example.com",  # type: ignore[arg-type]
            forward_headers=forward_headers,
        )
        adapter = PassthroughInferenceAdapter(config)
        spec = MagicMock()
        spec.provider_data_validator = _PROVIDER_VALIDATOR_PATH
        object.__setattr__(adapter, "__provider_spec__", spec)
        return adapter

    def test_no_auth_header_when_no_key_and_no_forward_auth(self):
        adapter = self._make_no_static_key_adapter(forward_headers={"tid": "X-Tenant-ID"})
        with RequestProviderDataContext({"tid": "acme"}):
            headers = adapter._build_request_headers()
        assert "Authorization" not in headers
        assert headers["X-Tenant-ID"] == "acme"

    def test_forwarded_auth_used_when_no_static_key(self):
        adapter = self._make_no_static_key_adapter(forward_headers={"tok": "Authorization"})
        with RequestProviderDataContext({"tok": "Bearer forwarded"}):
            headers = adapter._build_request_headers()
        assert headers["Authorization"] == "Bearer forwarded"

    def test_static_key_overwrites_forwarded_authorization(self):
        adapter = _make_adapter(forward_headers={"tok": "Authorization"}, api_key="sk-static")
        with RequestProviderDataContext({"tok": "Bearer forwarded"}):
            headers = adapter._build_request_headers()
        assert headers["Authorization"] == "Bearer sk-static"

    def test_empty_headers_when_nothing_configured(self):
        adapter = self._make_no_static_key_adapter(forward_headers=None)
        headers = adapter._build_request_headers()
        assert headers == {}


# ---------------------------------------------------------------------------
# PassthroughProviderDataValidator — optional fields
# ---------------------------------------------------------------------------


class TestPassthroughProviderDataValidator:
    def test_accepts_empty_payload(self):
        # existing configs that send neither field must still pass validation
        v = PassthroughProviderDataValidator()
        assert v.passthrough_url is None
        assert v.passthrough_api_key is None

    def test_accepts_only_passthrough_url(self):
        v = PassthroughProviderDataValidator(passthrough_url="http://x.example.com")
        assert str(v.passthrough_url) == "http://x.example.com/"
        assert v.passthrough_api_key is None

    def test_accepts_both_fields(self):
        v = PassthroughProviderDataValidator(
            passthrough_url="http://x.example.com",
            passthrough_api_key="sk-test",
        )
        assert str(v.passthrough_url) == "http://x.example.com/"
        assert v.passthrough_api_key is not None
        assert v.passthrough_api_key.get_secret_value() == "sk-test"

    def test_preserves_extra_fields_for_forwarding(self):
        # forward_headers source keys must be preserved so build_forwarded_headers can read them.
        v = PassthroughProviderDataValidator(maas_api_token="Bearer abc123")  # type: ignore[call-arg]
        assert v.passthrough_url is None
        assert v.passthrough_api_key is None
        assert v.model_dump().get("maas_api_token") == "Bearer abc123"


# ---------------------------------------------------------------------------
# PassthroughImplConfig.forward_headers validation
# ---------------------------------------------------------------------------


class TestPassthroughConfigValidation:
    def test_rejects_reserved_provider_data_keys(self):
        with pytest.raises(ValueError, match="reserved __ prefix"):
            PassthroughImplConfig(
                base_url="http://downstream.example.com",  # type: ignore[arg-type]
                forward_headers={"__authenticated_user": "X-User"},
            )

    def test_rejects_blocked_header_names(self):
        with pytest.raises(ValueError, match="blocked"):
            PassthroughImplConfig(
                base_url="http://downstream.example.com",  # type: ignore[arg-type]
                forward_headers={"maas_api_token": "Host"},
            )

    def test_accepts_safe_header_names(self):
        config = PassthroughImplConfig(
            base_url="http://downstream.example.com",  # type: ignore[arg-type]
            forward_headers={"maas_api_token": "Authorization", "tenant_id": "X-Tenant-ID"},
        )
        assert config.forward_headers == {"maas_api_token": "Authorization", "tenant_id": "X-Tenant-ID"}

    def test_rejects_operator_blocked_header_names(self):
        with pytest.raises(ValueError, match="blocked"):
            PassthroughImplConfig(
                base_url="http://downstream.example.com",  # type: ignore[arg-type]
                forward_headers={"trace_id": "X-Internal-Debug"},
                extra_blocked_headers=["x-internal-debug"],
            )

    def test_operator_blocked_names_are_case_insensitive(self):
        with pytest.raises(ValueError, match="blocked"):
            PassthroughImplConfig(
                base_url="http://downstream.example.com",  # type: ignore[arg-type]
                forward_headers={"trace_id": "X-INTERNAL-DEBUG"},
                extra_blocked_headers=["x-internal-debug"],
            )

    def test_rejects_empty_extra_blocked_header_names(self):
        with pytest.raises(ValueError, match="empty header name"):
            PassthroughImplConfig(
                base_url="http://downstream.example.com",  # type: ignore[arg-type]
                extra_blocked_headers=["   "],
            )

    def test_accepts_extra_blocked_headers_without_forward_headers(self):
        config = PassthroughImplConfig(
            base_url="http://downstream.example.com",  # type: ignore[arg-type]
            extra_blocked_headers=["x-internal-debug"],
        )
        assert config.extra_blocked_headers == ["x-internal-debug"]


# ---------------------------------------------------------------------------
# _get_passthrough_url — null check after fields made optional
# ---------------------------------------------------------------------------


class TestGetPassthroughUrl:
    def test_raises_when_base_url_and_passthrough_url_both_missing(self):
        config = PassthroughImplConfig()
        adapter = PassthroughInferenceAdapter(config)
        spec = MagicMock()
        spec.provider_data_validator = _PROVIDER_VALIDATOR_PATH
        object.__setattr__(adapter, "__provider_spec__", spec)

        with RequestProviderDataContext({}):
            with pytest.raises(ValueError, match="passthrough_url"):
                adapter._get_passthrough_url()

    def test_raises_when_no_provider_data_at_all(self):
        config = PassthroughImplConfig()
        adapter = PassthroughInferenceAdapter(config)
        spec = MagicMock()
        spec.provider_data_validator = _PROVIDER_VALIDATOR_PATH
        object.__setattr__(adapter, "__provider_spec__", spec)

        # no RequestProviderDataContext — PROVIDER_DATA_VAR is None
        with pytest.raises(ValueError, match="passthrough_url"):
            adapter._get_passthrough_url()


# ---------------------------------------------------------------------------
# Concurrent request isolation — ContextVar must not leak between tasks
# ---------------------------------------------------------------------------


class TestConcurrentRequestIsolation:
    async def test_contextvar_does_not_leak_between_parallel_requests(self):
        config = PassthroughImplConfig(
            base_url="http://downstream.example.com",  # type: ignore[arg-type]
            forward_headers={"token": "Authorization"},
        )
        adapter = PassthroughInferenceAdapter(config)
        spec = MagicMock()
        spec.provider_data_validator = _PROVIDER_VALIDATOR_PATH
        object.__setattr__(adapter, "__provider_spec__", spec)

        results: list[tuple[str, str | None]] = []

        async def run_with_data(token_value: str, expected: str) -> None:
            with RequestProviderDataContext({"token": token_value}):
                headers = adapter._build_request_headers()
                results.append((expected, headers.get("Authorization")))

        await asyncio.gather(
            run_with_data("Bearer token-A", "Bearer token-A"),
            run_with_data("Bearer token-B", "Bearer token-B"),
        )

        for expected, actual in results:
            assert actual == expected, f"ContextVar leaked: got {actual!r}, expected {expected!r}"


# ---------------------------------------------------------------------------
# Authorization case-collision — static api_key must win regardless of casing
# ---------------------------------------------------------------------------


class TestAuthorizationCaseCollision:
    def test_static_api_key_wins_over_lowercase_forwarded_authorization(self):
        """Static api_key must override a forwarded 'authorization' (lowercase) header.

        Without normalization, both end up as separate dict keys and httpx sends
        both Authorization values to the downstream — breaking the "static wins" guarantee.
        """
        adapter = _make_adapter(
            forward_headers={"user_token": "authorization"},  # lowercase — the collision case
            api_key="sk-static",
        )

        with RequestProviderDataContext({"user_token": "Bearer user-token"}):
            headers = adapter._build_request_headers()

        auth_keys = [k for k in headers if k.lower() == "authorization"]
        assert len(auth_keys) == 1, f"Expected 1 authorization header, got {auth_keys}"
        assert headers[auth_keys[0]] == "Bearer sk-static"

    def test_forwarded_authorization_used_when_no_static_key(self):
        """When no static api_key, forwarded Authorization reaches downstream."""
        config = PassthroughImplConfig(
            base_url="http://downstream.example.com",  # type: ignore[arg-type]
            forward_headers={"user_token": "Authorization"},
        )
        adapter = PassthroughInferenceAdapter(config)
        spec = MagicMock()
        spec.provider_data_validator = _PROVIDER_VALIDATOR_PATH
        object.__setattr__(adapter, "__provider_spec__", spec)

        with RequestProviderDataContext({"user_token": "Bearer user-token"}):
            headers = adapter._build_request_headers()

        assert headers.get("Authorization") == "Bearer user-token"


# ---------------------------------------------------------------------------
# Security hardening — Unicode line terminators and header value size limits
# ---------------------------------------------------------------------------


class TestHeaderValueSanitization:
    def test_strips_unicode_next_line(self):
        """U+0085 NEXT LINE must be stripped — some HTTP/1.0 proxies treat it as a line break."""
        result = build_forwarded_headers(
            {"token": "Bearer abc\x85suffix"},
            {"token": "Authorization"},
        )
        assert "\x85" not in result["Authorization"]
        assert result["Authorization"] == "Bearer abcsuffix"

    def test_strips_unicode_line_separator(self):
        """U+2028 LINE SEPARATOR must be stripped."""
        result = build_forwarded_headers(
            {"token": "Bearer abc\u2028suffix"},
            {"token": "Authorization"},
        )
        assert "\u2028" not in result["Authorization"]
        assert result["Authorization"] == "Bearer abcsuffix"

    def test_strips_unicode_paragraph_separator(self):
        """U+2029 PARAGRAPH SEPARATOR must be stripped."""
        result = build_forwarded_headers(
            {"token": "Bearer abc\u2029suffix"},
            {"token": "Authorization"},
        )
        assert "\u2029" not in result["Authorization"]
        assert result["Authorization"] == "Bearer abcsuffix"

    def test_oversized_value_dropped(self):
        """Values over 8 KB are silently dropped to prevent HTTP 431 from downstream."""
        big_value = "x" * 9000  # > 8192 bytes
        result = build_forwarded_headers(
            {"token": big_value, "tenant_id": "acme"},
            {"token": "Authorization", "tenant_id": "X-Tenant-ID"},
        )
        assert "Authorization" not in result
        assert result.get("X-Tenant-ID") == "acme"

    def test_exactly_8192_bytes_passes(self):
        """Values at exactly 8192 bytes are forwarded (boundary check)."""
        exact_value = "x" * 8192
        result = build_forwarded_headers(
            {"token": exact_value},
            {"token": "X-Custom"},
        )
        assert result.get("X-Custom") == exact_value
