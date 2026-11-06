# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from unittest.mock import MagicMock, patch

import pytest

from ogx.providers.utils.inference.openai_compat import (
    ToolCallArgumentsStreamFilter,
    get_stream_options_for_telemetry,
    normalize_tool_call_arguments,
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


class TestNormalizeToolCallArguments:
    """#6912/#6864: tool-call arguments translated to Anthropic's tool_use.input or
    Google's function-call args must always end up an object, matching those APIs.
    """

    def test_object_arguments_pass_through(self):
        assert normalize_tool_call_arguments('{"city": "Boston"}') == {"city": "Boston"}

    @pytest.mark.parametrize(
        "arguments",
        ["[1, 2]", '"hello"', "42", "true", "null"],
    )
    def test_non_object_json_values_become_empty_dict(self, arguments):
        assert normalize_tool_call_arguments(arguments) == {}

    def test_invalid_json_becomes_empty_dict(self):
        assert normalize_tool_call_arguments('{"city": "Bos') == {}

    @pytest.mark.parametrize("arguments", [None, ""])
    def test_empty_arguments_become_empty_dict(self, arguments):
        assert normalize_tool_call_arguments(arguments) == {}

    def test_logs_a_warning_without_the_raw_arguments(self, caplog):
        with caplog.at_level("WARNING"):
            normalize_tool_call_arguments("[1, 2]", tool_name="get_weather")

        assert "get_weather" in caplog.text
        assert "[1, 2]" not in caplog.text

    def test_no_warning_for_empty_arguments(self, caplog):
        with caplog.at_level("WARNING"):
            normalize_tool_call_arguments(None, tool_name="get_weather")

        assert caplog.text == ""


class TestToolCallArgumentsStreamFilter:
    """#6912: the streaming equivalent of normalize_tool_call_arguments, deciding from
    the first non-whitespace character instead of buffering the whole argument string.
    """

    def test_object_chunks_stream_through_unchanged(self):
        f = ToolCallArgumentsStreamFilter()

        assert f.feed('{"city":') == '{"city":'
        assert f.feed(' "Boston"}') == ' "Boston"}'
        assert f.finalize() is None

    @pytest.mark.parametrize("arguments", ["[1, 2]", '"hello"', "42", "true", "null"])
    def test_non_object_value_yields_a_single_empty_object_delta(self, arguments):
        f = ToolCallArgumentsStreamFilter()

        assert f.feed(arguments) == "{}"
        # Any further chunks for the same tool call must be dropped, not appended.
        assert f.feed("more text that must not leak through") is None
        assert f.finalize() is None

    def test_non_object_value_split_across_chunks_still_yields_one_empty_object(self):
        f = ToolCallArgumentsStreamFilter()

        assert f.feed('"he') == "{}"
        assert f.feed("llo") is None
        assert f.feed('"') is None

    def test_leading_whitespace_before_object_is_buffered_then_flushed(self):
        f = ToolCallArgumentsStreamFilter()

        assert f.feed("  ") is None
        assert f.feed('{"a": 1}') == '  {"a": 1}'

    def test_leading_whitespace_before_non_object_still_yields_empty_object(self):
        f = ToolCallArgumentsStreamFilter()

        assert f.feed("  ") is None
        assert f.feed("42") == "{}"

    def test_finalize_without_any_feed_yields_empty_object_once(self):
        f = ToolCallArgumentsStreamFilter()

        assert f.finalize() == "{}"
        # A second finalize (e.g. a defensive double-close) must not double-emit.
        assert f.finalize() is None

    def test_finalize_after_whitespace_only_feed_yields_empty_object(self):
        f = ToolCallArgumentsStreamFilter()

        assert f.feed("   ") is None
        assert f.finalize() == "{}"

    def test_finalize_after_object_decided_is_a_no_op(self):
        f = ToolCallArgumentsStreamFilter()

        f.feed("{}")
        assert f.finalize() is None

    def test_logs_a_warning_with_tool_name_without_the_raw_arguments(self, caplog):
        f = ToolCallArgumentsStreamFilter(tool_name="get_weather")

        with caplog.at_level("WARNING"):
            f.feed("[1, 2]")

        assert "get_weather" in caplog.text
        assert "[1, 2]" not in caplog.text
