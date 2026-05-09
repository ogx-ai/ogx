# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Tests for exception serialization/deserialization used in API recording replay."""

import httpx
import pytest
from ollama import ResponseError
from openai import (
    AuthenticationError,
    BadRequestError,
    InternalServerError,
    NotFoundError,
    PermissionDeniedError,
    RateLimitError,
    UnprocessableEntityError,
)
from openai import (
    ConflictError as OpenAIConflictError,
)

from ogx.core.exceptions.translation import translate_exception
from ogx.testing.exception_utils import (
    deserialize_exception,
    is_provider_sdk_exception,
    serialize_exception,
)
from ogx_api.common.errors import (
    BatchNotFoundError,
    ConflictError,
    ModelNotFoundError,
    OGXError,
)


def _openai_error(cls, status_code, body, message):
    """Construct an OpenAI error the way the SDK does internally."""
    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    response = httpx.Response(status_code, json=body or {}, request=request)
    return cls(message=message, response=response, body=body)


class TestSerializeException:
    """Test exception categorization and serialization for recording."""

    def test_ogx_error_serializes_as_ogx_category(self):
        exc = ModelNotFoundError("llama-3")
        data = serialize_exception(exc)
        assert data["category"] == "ogx"
        assert data["type"] == "ModelNotFoundError"
        assert "llama-3" in data["message"]
        assert data["status_code"] == 404

    def test_provider_sdk_exception_serializes_with_provider_and_body(self):
        body = {"error": {"code": "not_found"}}
        exc = _openai_error(NotFoundError, 404, body, "Batch not found")
        data = serialize_exception(exc)
        assert data["category"] == "provider_sdk"
        assert data["provider"] == "openai"
        assert data["status_code"] == 404
        assert data["body"] == body

    def test_provider_sdk_uses_error_attr_over_str(self):
        """Ollama's str(exc) appends '(status code: N)'; we serialize .error instead."""
        exc = ResponseError(error="model not found", status_code=404)
        data = serialize_exception(exc)
        assert data["message"] == "model not found"
        assert "(status code:" not in data["message"]

    def test_builtin_exception_serializes_by_type_name(self):
        exc = ValueError("invalid input")
        data = serialize_exception(exc)
        assert data["category"] == "builtin"
        assert data["type"] == "ValueError"
        assert data["message"] == "invalid input"

    def test_unknown_exception_serializes_with_type_and_message(self):
        exc = RuntimeError("unexpected failure")
        data = serialize_exception(exc)
        assert data["category"] == "unknown"
        assert data["type"] == "RuntimeError"
        assert data["message"] == "unexpected failure"


class TestDeserializeException:
    """Test exception reconstruction from recorded data."""

    def test_ogx_roundtrip_preserves_status_and_message(self):
        exc = BatchNotFoundError("batch-xyz")
        data = serialize_exception(exc)
        reconstructed = deserialize_exception(data)
        assert isinstance(reconstructed, OGXError)
        assert reconstructed.status_code == 404
        assert "batch-xyz" in str(reconstructed)

    def test_builtin_roundtrip_reconstructs_exact_type(self):
        exc = ValueError("bad value")
        data = serialize_exception(exc)
        reconstructed = deserialize_exception(data)
        assert type(reconstructed) is ValueError
        assert str(reconstructed) == "bad value"

    def test_unknown_roundtrip_falls_back_to_generic_exception(self):
        exc = RuntimeError("internal error")
        data = serialize_exception(exc)
        reconstructed = deserialize_exception(data)
        assert type(reconstructed) is Exception
        assert str(reconstructed) == "internal error"

    def test_deserialize_missing_category_defaults_to_unknown(self):
        data = {"type": "RuntimeError", "message": "legacy format"}
        reconstructed = deserialize_exception(data)
        assert type(reconstructed) is Exception
        assert str(reconstructed) == "legacy format"


class TestProviderSDKRoundTrip:
    """Full serialize -> deserialize round-trip for provider SDK exceptions.

    These test that the entire pipeline preserves the exception type, status code,
    and client-visible attributes across all mapped error types.
    """

    @pytest.mark.parametrize(
        "cls,status",
        [
            (BadRequestError, 400),
            (AuthenticationError, 401),
            (PermissionDeniedError, 403),
            (NotFoundError, 404),
            (OpenAIConflictError, 409),
            (UnprocessableEntityError, 422),
            (RateLimitError, 429),
            (InternalServerError, 500),
        ],
        ids=[
            "400-bad-request",
            "401-auth",
            "403-forbidden",
            "404-not-found",
            "409-conflict",
            "422-unprocessable",
            "429-rate-limit",
            "500-internal",
        ],
    )
    def test_openai_roundtrip_preserves_type_and_status(self, cls, status):
        original = _openai_error(cls, status, None, f"Error {status}")
        reconstructed = deserialize_exception(serialize_exception(original))
        assert type(reconstructed) is type(original)
        assert reconstructed.status_code == status

    def test_openai_roundtrip_preserves_parsed_attributes(self):
        """Body-derived attributes (.code, .type, .param) survive the round-trip."""
        body = {"message": "Invalid key", "type": "invalid_request_error", "code": "invalid_api_key", "param": None}
        original = _openai_error(AuthenticationError, 401, body, "Invalid key")
        reconstructed = deserialize_exception(serialize_exception(original))
        assert reconstructed.body == original.body
        assert reconstructed.code == original.code
        assert reconstructed.param == original.param
        assert reconstructed.type == original.type

    @pytest.mark.parametrize(
        "status,error_text",
        [
            (404, "model 'llama3' not found"),
            (500, "internal server error"),
        ],
        ids=["404-model-not-found", "500-internal"],
    )
    def test_ollama_roundtrip_preserves_type_status_and_error(self, status, error_text):
        original = ResponseError(error=error_text, status_code=status)
        reconstructed = deserialize_exception(serialize_exception(original))
        assert type(reconstructed) is type(original)
        assert reconstructed.status_code == status
        assert reconstructed.error == original.error


class TestReconstructedExceptionInterface:
    """Verify reconstructed exceptions work with server's translate_exception."""

    def test_ogx_reconstructed_translates_to_http(self):
        data = {"category": "ogx", "status_code": 404, "message": "Batch xyz not found"}
        exc = deserialize_exception(data)
        http_exc = translate_exception(exc)
        assert http_exc.status_code == 404
        assert "xyz" in http_exc.detail

    def test_provider_sdk_reconstructed_translates_to_http(self):
        data = {
            "category": "provider_sdk",
            "provider": "openai",
            "status_code": 429,
            "message": "Rate limit exceeded",
            "body": None,
        }
        exc = deserialize_exception(data)
        http_exc = translate_exception(exc)
        assert http_exc.status_code == 429

    def test_unknown_provider_uses_generic_but_preserves_status(self):
        data = {
            "category": "provider_sdk",
            "provider": "future_sdk",
            "status_code": 503,
            "message": "Service unavailable",
            "body": None,
        }
        exc = deserialize_exception(data)
        assert hasattr(exc, "status_code")
        assert exc.status_code == 503
        http_exc = translate_exception(exc)
        assert http_exc.status_code == 503


class TestIsProviderSdkException:
    """Test provider SDK exception detection used during serialization."""

    def test_openai_exception_detected(self):
        request = httpx.Request("GET", "https://api.openai.com/v1/models")
        response = httpx.Response(404, request=request)
        assert is_provider_sdk_exception(NotFoundError(message="x", response=response, body=None))

    def test_ollama_exception_detected(self):
        assert is_provider_sdk_exception(ResponseError(error="x", status_code=404))

    def test_ogx_not_detected_as_provider_sdk(self):
        """OGXError has status_code but is handled separately (ogx category)."""
        exc = ConflictError("conflict")
        data = serialize_exception(exc)
        assert data["category"] == "ogx"

    def test_plain_exception_not_detected(self):
        assert not is_provider_sdk_exception(ValueError("x"))
