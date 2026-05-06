# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""
Tests for parse_data_url in ogx.providers.utils.common.data_url.

Categories:
  - Valid URLs: plain data, base64, charset, combined modifiers, complex mime types, multiline data
  - Invalid URLs: wrong scheme, missing comma, empty string, missing mime type
"""

import pytest

from ogx.providers.utils.common.data_url import parse_data_url


class TestParseDataUrlValid:
    def test_plain_text(self):
        parts = parse_data_url("data:text/plain,hello world")
        assert parts["mimetype"] == "text/plain"
        assert parts["data"] == "hello world"
        assert parts["is_base64"] is False
        assert parts["encoding"] is None

    def test_base64_flag(self):
        parts = parse_data_url("data:text/plain;base64,aGVsbG8=")
        assert parts["mimetype"] == "text/plain"
        assert parts["data"] == "aGVsbG8="
        assert parts["is_base64"] is True
        assert parts["base64"] == ";base64"

    def test_charset_only(self):
        parts = parse_data_url("data:text/plain;charset=utf-8,hello")
        assert parts["mimetype"] == "text/plain"
        assert parts["encoding"] == "utf-8"
        assert parts["charset"] == ";charset=utf-8"
        assert parts["is_base64"] is False
        assert parts["data"] == "hello"

    def test_charset_and_base64(self):
        parts = parse_data_url("data:text/plain;charset=utf-8;base64,aGVsbG8=")
        assert parts["mimetype"] == "text/plain"
        assert parts["encoding"] == "utf-8"
        assert parts["is_base64"] is True
        assert parts["data"] == "aGVsbG8="

    def test_image_mime_type(self):
        parts = parse_data_url("data:image/png;base64,abc123==")
        assert parts["mimetype"] == "image/png"
        assert parts["is_base64"] is True
        assert parts["data"] == "abc123=="

    def test_mime_type_with_plus(self):
        parts = parse_data_url("data:application/json+xml,{}")
        assert parts["mimetype"] == "application/json+xml"
        assert parts["data"] == "{}"
        assert parts["is_base64"] is False

    def test_empty_data_portion(self):
        parts = parse_data_url("data:text/plain,")
        assert parts["mimetype"] == "text/plain"
        assert parts["data"] == ""
        assert parts["is_base64"] is False

    def test_multiline_data(self):
        # re.DOTALL allows newlines inside the data portion
        parts = parse_data_url("data:text/plain,line1\nline2")
        assert parts["data"] == "line1\nline2"

    def test_hyphenated_charset(self):
        parts = parse_data_url("data:text/html;charset=ISO-8859-1,<b>hi</b>")
        assert parts["encoding"] == "ISO-8859-1"

    def test_mime_type_with_dot(self):
        parts = parse_data_url("data:application/vnd.ms-excel,data")
        assert parts["mimetype"] == "application/vnd.ms-excel"


class TestParseDataUrlInvalid:
    def test_http_url_raises(self):
        with pytest.raises(ValueError, match="Invalid Data URL format"):
            parse_data_url("http://example.com/image.png")

    def test_missing_comma_raises(self):
        with pytest.raises(ValueError, match="Invalid Data URL format"):
            parse_data_url("data:text/plain")

    def test_empty_string_raises(self):
        with pytest.raises(ValueError, match="Invalid Data URL format"):
            parse_data_url("")

    def test_missing_mime_type_raises(self):
        # mimetype pattern requires at least one character
        with pytest.raises(ValueError, match="Invalid Data URL format"):
            parse_data_url("data:,hello")

    def test_plain_string_raises(self):
        with pytest.raises(ValueError, match="Invalid Data URL format"):
            parse_data_url("not a data url at all")
