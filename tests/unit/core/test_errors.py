# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


import httpx
from fastapi import HTTPException
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, ValidationError

from ogx.core.exceptions.mapping import translate_exception_to_http
from ogx.core.exceptions.translation import translate_exception
from ogx_api.common.errors import (
    BatchNotFoundError,
    ClientListCommand,
    ConflictError,
    ConversationItemNotFoundError,
    ConversationNotFoundError,
    InternalServerError,
    InvalidParameterError,
    ModelNotFoundError,
    ModelTypeError,
    OGXError,
    ResourceNotFoundError,
    ResponseInputItemNotFoundError,
    ResponseNotFoundError,
    ServiceNotEnabledError,
    TokenValidationError,
    UnsupportedModelError,
)


class TestClientListCommand:
    def test_basic_command(self):
        cmd = ClientListCommand("files.list")
        assert str(cmd) == "Use 'client.files.list()'."

    def test_with_argument(self):
        cmd = ClientListCommand("connectors.list_tools", "my-connector")
        assert str(cmd) == "Use 'client.connectors.list_tools(\"my-connector\")'."

    def test_with_multiple_arguments(self):
        cmd = ClientListCommand("widgets.find", ["arg1", "arg2"])
        assert str(cmd) == 'Use \'client.widgets.find("arg1", "arg2")\'.'

    def test_with_resource_plural(self):
        cmd = ClientListCommand("batches.list", resource_name_plural="batches")
        assert str(cmd) == "Use 'client.batches.list()' to list available batches."


class TestTranslateExceptionToHttp:
    """Tests for translate_exception_to_http which walks the MRO to find
    the first mapped exception type in EXCEPTION_MAP."""

    # ── Direct matches for each mapped type ──────────────────────────

    def test_value_error(self):
        exc = ValueError("bad input")
        result = translate_exception_to_http(exc)
        assert result is not None
        assert result.status_code == httpx.codes.BAD_REQUEST
        assert "bad input" in result.detail

    def test_permission_error(self):
        exc = PermissionError("access denied")
        result = translate_exception_to_http(exc)
        assert result is not None
        assert result.status_code == httpx.codes.FORBIDDEN
        assert "access denied" in result.detail

    def test_connection_error(self):
        exc = ConnectionError("connection refused")
        result = translate_exception_to_http(exc)
        assert result is not None
        assert result.status_code == httpx.codes.BAD_GATEWAY

    def test_timeout_error(self):
        exc = TimeoutError("timed out")
        result = translate_exception_to_http(exc)
        assert result is not None
        assert result.status_code == httpx.codes.GATEWAY_TIMEOUT
        assert "timed out" in result.detail

    def test_asyncio_timeout_error(self):
        exc = TimeoutError("async timed out")
        result = translate_exception_to_http(exc)
        assert result is not None
        assert result.status_code == httpx.codes.GATEWAY_TIMEOUT

    def test_not_implemented_error(self):
        exc = NotImplementedError("not supported")
        result = translate_exception_to_http(exc)
        assert result is not None
        assert result.status_code == httpx.codes.NOT_IMPLEMENTED
        assert "not supported" in result.detail

    # ── Subclass matching via MRO ────────────────────────────────────

    def test_subclass_one_level_deep(self):
        """A direct subclass of a mapped type should match via MRO."""

        class CustomValueError(ValueError):
            pass

        exc = CustomValueError("custom")
        result = translate_exception_to_http(exc)
        assert result is not None
        assert result.status_code == httpx.codes.BAD_REQUEST
        assert "custom" in result.detail

    def test_subclass_two_levels_deep(self):
        """A two-level deep subclass of a mapped type should still match."""

        class SpecificValueError(ValueError):
            pass

        class VerySpecificValueError(SpecificValueError):
            pass

        exc = VerySpecificValueError("deep")
        result = translate_exception_to_http(exc)
        assert result is not None
        assert result.status_code == httpx.codes.BAD_REQUEST

    def test_ogx_error_not_in_map(self):
        """OGXError subclasses (single inheritance) are not in
        EXCEPTION_MAP and should return None from translate_exception_to_http.
        They are handled separately by translate_exception."""
        exc = ResourceNotFoundError("abc", "Widget")
        assert translate_exception_to_http(exc) is None

        exc2 = ModelNotFoundError("llama-3")
        assert translate_exception_to_http(exc2) is None

    # ── Multiple mapped parents: MRO order determines winner ─────────

    def test_multiple_mapped_parents_first_wins(self):
        """When a class inherits from two mapped types, MRO puts the
        first parent before the second, so it should win."""

        class PermThenValueError(PermissionError, ValueError):
            pass

        exc = PermThenValueError("denied")
        result = translate_exception_to_http(exc)
        assert result is not None
        # PermissionError comes first in MRO → 403 FORBIDDEN
        assert result.status_code == httpx.codes.FORBIDDEN

    def test_multiple_mapped_parents_reversed_order(self):
        """Reversing the parent order flips which mapped type wins."""

        class ValueThenPermError(ValueError, PermissionError):
            pass

        exc = ValueThenPermError("invalid")
        result = translate_exception_to_http(exc)
        assert result is not None
        # ValueError comes first in MRO → 400 BAD_REQUEST
        assert result.status_code == httpx.codes.BAD_REQUEST

    # ── Unmapped types before mapped type in the MRO ─────────────────

    def test_unmapped_ancestors_before_mapped(self):
        """KeyError -> LookupError are not in the map. For
        class Err(KeyError, ValueError), MRO is:
        Err -> KeyError -> LookupError -> ValueError -> Exception
        so it should skip unmapped types and match ValueError."""

        class LookupAndValueError(KeyError, ValueError):
            pass

        exc = LookupAndValueError("missing")
        result = translate_exception_to_http(exc)
        assert result is not None
        assert result.status_code == httpx.codes.BAD_REQUEST

    # ── No match returns None ────────────────────────────────────────

    def test_unmapped_runtime_error(self):
        exc = RuntimeError("boom")
        assert translate_exception_to_http(exc) is None

    def test_unmapped_key_error(self):
        exc = KeyError("missing")
        assert translate_exception_to_http(exc) is None

    def test_bare_exception(self):
        exc = Exception("generic")
        assert translate_exception_to_http(exc) is None

    def test_ogx_error_base_not_in_map(self):
        """OGXError inherits from Exception which is NOT in the
        map. The base class itself should not match since it's handled
        separately by translate_exception via its status_code attr."""

        class BareStackError(OGXError):
            status_code = httpx.codes.IM_A_TEAPOT

        exc = BareStackError("teapot")
        assert translate_exception_to_http(exc) is None

    # ── Detail uses exception message, fallback when empty ──────────

    def test_detail_uses_exception_message(self):
        """When the exception has a message, it is used as the detail."""
        exc = ValueError("price must be positive")
        result = translate_exception_to_http(exc)
        assert result is not None
        assert result.detail == "price must be positive"

    def test_empty_message_uses_fallback(self):
        """When the exception message is empty, the fallback from the
        map is used instead."""
        exc = ValueError("")
        result = translate_exception_to_http(exc)
        assert result is not None
        assert result.detail == "Invalid value"

    def test_connection_error_uses_message(self):
        exc = ConnectionError("connection refused")
        result = translate_exception_to_http(exc)
        assert result is not None
        assert result.detail == "connection refused"

    def test_connection_error_empty_uses_fallback(self):
        exc = ConnectionError("")
        result = translate_exception_to_http(exc)
        assert result is not None
        assert result.detail == "Connection error"

    def test_permission_error_uses_message(self):
        exc = PermissionError("read-only resource")
        result = translate_exception_to_http(exc)
        assert result is not None
        assert result.detail == "read-only resource"

    def test_timeout_uses_message(self):
        exc = TimeoutError("after 30s")
        result = translate_exception_to_http(exc)
        assert result is not None
        assert result.detail == "after 30s"

    def test_timeout_empty_uses_fallback(self):
        exc = TimeoutError()
        result = translate_exception_to_http(exc)
        assert result is not None
        assert result.detail == "Operation timed out"


class TestTranslateException:
    """Tests for translate_exception, the top-level exception handler
    registered with FastAPI. Validates that every exception type always
    produces an HTTPException with a logically correct status code."""

    # ── Always returns HTTPException ─────────────────────────────────

    def test_always_returns_http_exception(self):
        """No matter what exception goes in, an HTTPException must come out."""
        exceptions = [
            ValueError("bad"),
            RuntimeError("boom"),
            Exception("generic"),
            TypeError("wrong type"),
            KeyError("missing"),
            PermissionError("denied"),
            NotImplementedError("todo"),
            ModelNotFoundError("llama-3"),
            ConflictError("already exists"),
        ]
        for exc in exceptions:
            result = translate_exception(exc)
            assert isinstance(result, HTTPException), (
                f"translate_exception({type(exc).__name__}) did not return HTTPException"
            )

    # ── OGXError uses its own status_code, NOT EXCEPTION_MAP ──

    def test_resource_not_found_error_uses_404(self):
        """ResourceNotFoundError(OGXError) has status_code 404."""
        exc = ResourceNotFoundError("abc", "Widget")
        result = translate_exception(exc)
        assert result.status_code == httpx.codes.NOT_FOUND

    def test_model_not_found_error_uses_404(self):
        """ModelNotFoundError -> ResourceNotFoundError -> OGXError.
        Three levels deep, should still get 404."""
        exc = ModelNotFoundError("gpt-missing")
        result = translate_exception(exc)
        assert result.status_code == httpx.codes.NOT_FOUND

    def test_batch_not_found_error_uses_404(self):
        exc = BatchNotFoundError("batch-123")
        result = translate_exception(exc)
        assert result.status_code == httpx.codes.NOT_FOUND

    def test_conflict_error_uses_409(self):
        """ConflictError(OGXError) has status_code 409 CONFLICT."""
        exc = ConflictError("resource already exists")
        result = translate_exception(exc)
        assert result.status_code == httpx.codes.CONFLICT

    def test_token_validation_error_uses_401(self):
        """TokenValidationError(OGXError) has status_code 401 UNAUTHORIZED."""
        exc = TokenValidationError("expired token")
        result = translate_exception(exc)
        assert result.status_code == httpx.codes.UNAUTHORIZED

    def test_model_type_error_uses_400(self):
        """ModelTypeError(OGXError) has status_code 400 BAD_REQUEST."""
        exc = ModelTypeError("llama-3", "embedding", "llm")
        result = translate_exception(exc)
        assert result.status_code == httpx.codes.BAD_REQUEST

    def test_unsupported_model_error_uses_400(self):
        exc = UnsupportedModelError("bad-model", ["llama-3", "gpt-4"])
        result = translate_exception(exc)
        assert result.status_code == httpx.codes.BAD_REQUEST

    def test_invalid_parameter_error_uses_400(self):
        exc = InvalidParameterError("max_tool_calls", 0, "Must be >= 1.")
        result = translate_exception(exc)
        assert result.status_code == httpx.codes.BAD_REQUEST

    def test_service_not_enabled_error_uses_503(self):
        exc = ServiceNotEnabledError("Safety API")
        result = translate_exception(exc)
        assert result.status_code == httpx.codes.SERVICE_UNAVAILABLE

    def test_internal_server_error_uses_500(self):
        exc = InternalServerError()
        result = translate_exception(exc)
        assert result.status_code == httpx.codes.INTERNAL_SERVER_ERROR

    def test_response_not_found_error_uses_404(self):
        exc = ResponseNotFoundError("resp_abc123")
        result = translate_exception(exc)
        assert result.status_code == httpx.codes.NOT_FOUND

    def test_response_input_item_not_found_error_uses_404(self):
        exc = ResponseInputItemNotFoundError("input_abc", "resp_xyz")
        result = translate_exception(exc)
        assert result.status_code == httpx.codes.NOT_FOUND

    def test_conversation_not_found_error_uses_404(self):
        exc = ConversationNotFoundError("conv_nonexistent")
        result = translate_exception(exc)
        assert result.status_code == httpx.codes.NOT_FOUND

    def test_conversation_item_not_found_error_uses_404(self):
        exc = ConversationItemNotFoundError("msg_abc123", "conv_xyz789")
        result = translate_exception(exc)
        assert result.status_code == httpx.codes.NOT_FOUND

    def test_ogx_error_preserves_message(self):
        exc = ModelNotFoundError("llama-3")
        result = translate_exception(exc)
        assert "llama-3" in result.detail
        assert "not found" in result.detail.lower()

    # ── Mapped exceptions (non-OGXError) ─────────────────────

    def test_plain_value_error_maps_to_400(self):
        exc = ValueError("invalid input")
        result = translate_exception(exc)
        assert result.status_code == httpx.codes.BAD_REQUEST

    def test_permission_error_maps_to_403(self):
        exc = PermissionError("denied")
        result = translate_exception(exc)
        assert result.status_code == httpx.codes.FORBIDDEN

    def test_not_implemented_error_maps_to_501(self):
        exc = NotImplementedError("coming soon")
        result = translate_exception(exc)
        assert result.status_code == httpx.codes.NOT_IMPLEMENTED

    def test_timeout_error_maps_to_504(self):
        exc = TimeoutError("timed out")
        result = translate_exception(exc)
        assert result.status_code == httpx.codes.GATEWAY_TIMEOUT

    def test_connection_error_maps_to_502(self):
        exc = ConnectionError("refused")
        result = translate_exception(exc)
        assert result.status_code == httpx.codes.BAD_GATEWAY

    # ── Provider SDK exceptions (duck-typed status_code) ─────────────

    def test_provider_sdk_exception_preserves_status_code(self):
        """Exceptions with an int status_code attr (like OpenAI SDK errors)
        should have that status code preserved."""

        class FakeProviderError(Exception):
            def __init__(self, message, status_code):
                super().__init__(message)
                self.status_code = status_code

        exc = FakeProviderError("rate limited", 429)
        result = translate_exception(exc)
        assert result.status_code == 429

    def test_provider_sdk_exception_with_401(self):
        class FakeAuthError(Exception):
            def __init__(self):
                super().__init__("invalid api key")
                self.status_code = 401

        exc = FakeAuthError()
        result = translate_exception(exc)
        assert result.status_code == 401

    # ── Completely unknown exceptions → 500 ──────────────────────────

    def test_runtime_error_falls_to_500(self):
        exc = RuntimeError("unexpected")
        result = translate_exception(exc)
        assert result.status_code == httpx.codes.INTERNAL_SERVER_ERROR

    def test_bare_exception_falls_to_500(self):
        exc = Exception("unknown")
        result = translate_exception(exc)
        assert result.status_code == httpx.codes.INTERNAL_SERVER_ERROR

    def test_key_error_falls_to_500(self):
        exc = KeyError("missing_key")
        result = translate_exception(exc)
        assert result.status_code == httpx.codes.INTERNAL_SERVER_ERROR

    def test_unknown_exception_detail_is_generic(self):
        """Unknown exceptions should NOT leak internal details."""
        exc = RuntimeError("secret internal state: db_password=hunter2")
        result = translate_exception(exc)
        assert "hunter2" not in result.detail
        assert "internal" in result.detail.lower()

    # ── ValidationError / RequestValidationError → 400 ───────────────

    def test_request_validation_error(self):
        exc = RequestValidationError(errors=[{"loc": ("body", "name"), "msg": "field required", "type": "missing"}])
        result = translate_exception(exc)
        assert result.status_code == httpx.codes.BAD_REQUEST
        assert isinstance(result.detail, dict)
        assert "errors" in result.detail

    def test_pydantic_validation_error(self):
        """Pydantic ValidationError should be wrapped into
        RequestValidationError and return 400."""

        class StrictModel(BaseModel):
            name: str
            age: int

        try:
            StrictModel(name=123, age="not_a_number")  # type: ignore[arg-type]
        except ValidationError as exc:
            result = translate_exception(exc)
            assert result.status_code == httpx.codes.BAD_REQUEST
            assert isinstance(result.detail, dict)
            assert "errors" in result.detail
            assert len(result.detail["errors"]) > 0
