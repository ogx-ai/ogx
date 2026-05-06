# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from unittest.mock import MagicMock, patch

from ogx.providers.utils.inference.openai_compat import (
    get_stream_options_for_telemetry,
)


class TestGetStreamOptionsForTelemetry:
    def test_returns_original_when_not_streaming(self):
        stream_options = {"keep": True}

        result = get_stream_options_for_telemetry(stream_options, False)

        assert result is stream_options

    def test_streaming_without_active_span_returns_original(self):
        stream_options = {"keep": True}

        with patch("opentelemetry.trace.get_current_span", return_value=None):
            result = get_stream_options_for_telemetry(stream_options, True)

        assert result is stream_options

    def test_streaming_with_inactive_span_returns_original(self):
        stream_options = {"keep": True}
        mock_span = MagicMock()
        mock_span.is_recording.return_value = False

        with patch("opentelemetry.trace.get_current_span", return_value=mock_span):
            result = get_stream_options_for_telemetry(stream_options, True)

        assert result is stream_options

    def test_streaming_with_active_span_adds_usage_when_missing(self):
        mock_span = MagicMock()
        mock_span.is_recording.return_value = True

        with patch("opentelemetry.trace.get_current_span", return_value=mock_span):
            result = get_stream_options_for_telemetry(None, True)

        assert result == {"include_usage": True}

    def test_streaming_with_active_span_merges_existing_options(self):
        stream_options = {"other_option": "value"}
        mock_span = MagicMock()
        mock_span.is_recording.return_value = True

        with patch("opentelemetry.trace.get_current_span", return_value=mock_span):
            result = get_stream_options_for_telemetry(stream_options, True)

        assert result == {"other_option": "value", "include_usage": True}
        assert stream_options == {"other_option": "value"}

    def test_streaming_with_active_span_overrides_include_usage_false(self):
        stream_options = {"include_usage": False}
        mock_span = MagicMock()
        mock_span.is_recording.return_value = True

        with patch("opentelemetry.trace.get_current_span", return_value=mock_span):
            result = get_stream_options_for_telemetry(stream_options, True)

        assert result["include_usage"] is True
