# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from unittest.mock import patch

import pytest

from ogx.providers.utils.memory.vector_store import (
    content_from_data_and_mime_type,
    validate_tiktoken_encoding,
)


def test_validate_tiktoken_encoding_succeeds():
    """validate_tiktoken_encoding does not raise when tiktoken is available."""
    validate_tiktoken_encoding("cl100k_base")


def test_validate_tiktoken_encoding_raises_on_failure():
    """validate_tiktoken_encoding raises RuntimeError with operator context on failure."""
    import ogx.providers.utils.memory.vector_store as vs_module

    with patch.object(vs_module, "_get_encoding", side_effect=RuntimeError("encoding unavailable")):
        with pytest.raises(RuntimeError, match="encoding unavailable"):
            validate_tiktoken_encoding("cl100k_base")


def test_content_from_data_and_mime_type_success_utf8():
    """Test successful decoding with UTF-8 encoding."""
    data = "Hello World! 🌍".encode()
    mime_type = "text/plain"

    with patch("chardet.detect") as mock_detect:
        mock_detect.return_value = {"encoding": "utf-8"}

        result = content_from_data_and_mime_type(data, mime_type)

        mock_detect.assert_called_once_with(data)
        assert result == "Hello World! 🌍"


def test_content_from_data_and_mime_type_error_win1252():
    """Test fallback to UTF-8 when Windows-1252 encoding detection fails."""
    data = "Hello World! 🌍".encode()
    mime_type = "text/plain"

    with patch("chardet.detect") as mock_detect:
        mock_detect.return_value = {"encoding": "Windows-1252"}

        result = content_from_data_and_mime_type(data, mime_type)

        assert result == "Hello World! 🌍"
        mock_detect.assert_called_once_with(data)


def test_content_from_data_and_mime_type_both_encodings_fail():
    """Test that exceptions are raised when both primary and UTF-8 encodings fail."""
    # Create invalid byte sequence that fails with both encodings
    data = b"\xff\xfe\x00\x8f"  # Invalid UTF-8 sequence
    mime_type = "text/plain"

    with patch("chardet.detect") as mock_detect:
        mock_detect.return_value = {"encoding": "windows-1252"}

        # Should raise an exception instead of returning empty string
        with pytest.raises(UnicodeDecodeError):
            content_from_data_and_mime_type(data, mime_type)
