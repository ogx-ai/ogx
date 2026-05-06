# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Tests for provider registry and exception reconstruction dispatch."""

import types
from unittest.mock import patch

import httpx
import pytest
from ollama import ResponseError
from openai import NotFoundError

from ogx.testing.exception_utils import GenericProviderError
from ogx.testing.providers import (
    PROVIDERS,
    ProviderConfig,
    create_provider_error,
    detect_provider,
)
from ogx.testing.providers._config import _validate_provider


class TestDetectProvider:
    """Test provider detection from exception module path."""

    def test_openai_exception_detected(self):
        request = httpx.Request("GET", "https://api.openai.com/v1/models")
        response = httpx.Response(404, request=request)
        exc = NotFoundError(message="x", response=response, body=None)
        assert detect_provider(exc) == "openai"

    def test_ollama_exception_detected(self):
        exc = ResponseError(error="model not found", status_code=404)
        assert detect_provider(exc) == "ollama"

    def test_unknown_exception_returns_unknown(self):
        exc = ValueError("plain Python exception")
        assert detect_provider(exc) == "unknown"


class TestCreateProviderError:
    """Test provider-specific error reconstruction."""

    def test_openai_reconstructs_specific_type_by_status(self):
        """404 -> NotFoundError, 429 -> RateLimitError, etc."""
        exc = create_provider_error("openai", 404, {"error": {"code": "not_found"}}, "Not found")
        assert isinstance(exc, NotFoundError)
        assert exc.status_code == 404
        assert "not found" in str(exc).lower()

    def test_openai_unknown_status_falls_back_to_api_status_error(self):
        """Unmapped status codes still produce valid APIStatusError."""
        exc = create_provider_error("openai", 418, None, "I'm a teapot")
        assert exc.status_code == 418
        assert "teapot" in str(exc).lower()

    def test_ollama_reconstructs_response_error(self):
        exc = create_provider_error("ollama", 404, None, "model not found")
        assert isinstance(exc, ResponseError)
        assert exc.status_code == 404
        assert "not found" in str(exc).lower()

    def test_unknown_provider_returns_generic_with_status_and_body(self):
        """Unknown providers get GenericProviderError for consistent replay."""
        exc = create_provider_error("future_sdk", 503, {"retry_after": 60}, "Unavailable")
        assert isinstance(exc, GenericProviderError)
        assert exc.status_code == 503
        assert exc.body == {"retry_after": 60}
        assert "unavailable" in str(exc).lower()


class TestProviderRegistration:
    """Verify adding a provider to the registry makes it work for detect and create."""

    @staticmethod
    def _fake_sdk_module(name: str = "example_sdk") -> types.ModuleType:
        """Create a minimal fake SDK module for testing."""
        return types.ModuleType(name)

    def test_detect_provider_uses_registered_providers(self):
        """Exception from a registered SDK module is detected."""
        exc_cls = type("APIError", (Exception,), {"__module__": "example_sdk.errors"})

        fake_config = ProviderConfig(
            name="example",
            sdk_module=self._fake_sdk_module("example_sdk"),
            create_error=lambda s, b, m: exc_cls(m),
        )
        with patch.dict(PROVIDERS, {"example": fake_config}, clear=False):
            exc = exc_cls("test")
            assert detect_provider(exc) == "example"

    def test_create_provider_error_uses_registered_providers(self):
        """Registered provider's create_error is called."""

        class CustomError(Exception):
            status_code = 0
            body = None

        def create_error(status_code: int, body, message: str):
            exc = CustomError(message)
            exc.status_code = status_code
            exc.body = body
            return exc

        fake_config = ProviderConfig(
            name="example",
            sdk_module=self._fake_sdk_module("example_sdk"),
            create_error=create_error,
        )
        with patch.dict(PROVIDERS, {"example": fake_config}, clear=False):
            exc = create_provider_error("example", 503, {"retry": 60}, "Down")
            assert isinstance(exc, CustomError)
            assert exc.status_code == 503
            assert exc.body == {"retry": 60}


class TestProviderValidation:
    """Verify ProviderConfig validation fails fast with clear errors."""

    @staticmethod
    def _fake_sdk_module(name: str = "example_sdk") -> types.ModuleType:
        return types.ModuleType(name)

    def test_valid_config_passes(self):
        """Valid ProviderConfig does not raise."""
        config = ProviderConfig(
            name="example",
            sdk_module=self._fake_sdk_module(),
            create_error=lambda s, b, m: Exception(m),
        )
        _validate_provider(config)

    def test_empty_name_raises(self):
        """Empty name raises with clear message."""
        config = ProviderConfig(name="", sdk_module=self._fake_sdk_module(), create_error=lambda *a: Exception())
        with pytest.raises(ValueError, match="name must be a non-empty str"):
            _validate_provider(config)

    def test_invalid_name_type_raises(self):
        """Non-str name raises."""
        config = ProviderConfig(name=123, sdk_module=self._fake_sdk_module(), create_error=lambda *a: Exception())
        with pytest.raises(ValueError, match="name"):
            _validate_provider(config)

    def test_sdk_module_must_be_a_module(self):
        """sdk_module must be an actual module, not a class or string."""
        config = ProviderConfig(name="x", sdk_module="not_a_module", create_error=lambda *a: Exception())
        with pytest.raises(ValueError, match="sdk_module must be a module"):
            _validate_provider(config)

    def test_non_callable_create_error_raises(self):
        """create_error must be callable."""
        config = ProviderConfig(name="x", sdk_module=self._fake_sdk_module(), create_error="not a function")
        with pytest.raises(ValueError, match="create_error must be callable"):
            _validate_provider(config)
