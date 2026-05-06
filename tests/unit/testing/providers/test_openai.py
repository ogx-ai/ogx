# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Tests for OpenAI provider exception reconstruction."""

import pytest
from openai import (
    APIStatusError,
    AuthenticationError,
    BadRequestError,
    ConflictError,
    InternalServerError,
    NotFoundError,
    PermissionDeniedError,
    RateLimitError,
    UnprocessableEntityError,
)

from ogx.testing.providers.openai import create_error


class TestOpenAICreateError:
    """Test OpenAI-specific error reconstruction for replay."""

    @pytest.mark.parametrize(
        "status,expected_class",
        [
            (400, BadRequestError),
            (401, AuthenticationError),
            (403, PermissionDeniedError),
            (404, NotFoundError),
            (409, ConflictError),
            (422, UnprocessableEntityError),
            (429, RateLimitError),
            (500, InternalServerError),
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
    def test_status_code_maps_to_correct_class(self, status, expected_class):
        """Each mapped status code reconstructs to the expected OpenAI error type."""
        exc = create_error(status, None, f"error {status}")
        assert isinstance(exc, expected_class)
        assert exc.status_code == status

    def test_body_preserved_with_parsed_attributes(self):
        """Body is attached and the SDK parses .code, .type, .param from it."""
        body = {"code": "invalid_api_key", "type": "invalid_request_error", "param": "model", "message": "bad key"}
        exc = create_error(401, body, "Invalid API key")
        assert exc.body == body
        assert exc.code == "invalid_api_key"
        assert exc.type == "invalid_request_error"
        assert exc.param == "model"

    def test_unmapped_status_uses_api_status_error_base(self):
        """Status codes without specific mapping still produce valid API errors."""
        exc = create_error(503, None, "Service unavailable")
        assert isinstance(exc, APIStatusError)
        assert exc.status_code == 503
