# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Tests for Ollama provider exception reconstruction."""

from ollama import ResponseError

from ogx.testing.providers.ollama import create_error


class TestOllamaCreateError:
    """Test Ollama-specific error reconstruction for replay."""

    def test_reconstructs_response_error_with_status_and_message(self):
        """Ollama errors replay with status_code for translate_exception compatibility."""
        exc = create_error(404, None, "model not found")
        assert isinstance(exc, ResponseError)
        assert exc.status_code == 404
        assert "not found" in str(exc).lower()

    def test_error_attribute_matches_input(self):
        """The .error attribute stores the raw error text for client inspection."""
        exc = create_error(404, None, "model 'llama3' not found")
        assert exc.error == "model 'llama3' not found"

    def test_body_parameter_ignored_ollama_has_no_body_attr(self):
        """Ollama ResponseError doesn't have body; create_error accepts but doesn't use it."""
        exc = create_error(500, {"detail": "internal"}, "Internal server error")
        assert isinstance(exc, ResponseError)
        assert exc.status_code == 500
