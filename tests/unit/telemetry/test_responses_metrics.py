# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Unit tests for responses API metrics (tool types and agentic calls)."""

from unittest.mock import MagicMock, patch

from ogx.providers.inline.responses.builtin.impl import (
    _agentic_calls_total,
    _record_tool_usage,
    _tool_types_used_total,
)


class TestResponsesMetricInstruments:
    """Test that metric instruments are properly defined."""

    def test_tool_types_used_total_exists(self):
        assert _tool_types_used_total is not None
        assert hasattr(_tool_types_used_total, "add")

    def test_agentic_calls_total_exists(self):
        assert _agentic_calls_total is not None
        assert hasattr(_agentic_calls_total, "add")

    def test_counters_can_record(self):
        _tool_types_used_total.add(1, {"tool_type": "function"})
        _agentic_calls_total.add(1)


class TestResponsesMetricsConstants:
    """Test that metric constants follow naming conventions."""

    def test_metric_names_follow_convention(self):
        from ogx.telemetry.constants import (
            RESPONSES_AGENTIC_CALLS_TOTAL,
            RESPONSES_TOOL_TYPES_USED_TOTAL,
        )

        assert RESPONSES_TOOL_TYPES_USED_TOTAL.startswith("ogx.")
        assert RESPONSES_AGENTIC_CALLS_TOTAL.startswith("ogx.")

        assert "responses" in RESPONSES_TOOL_TYPES_USED_TOTAL
        assert "responses" in RESPONSES_AGENTIC_CALLS_TOTAL

        assert RESPONSES_TOOL_TYPES_USED_TOTAL.endswith("_total")
        assert RESPONSES_AGENTIC_CALLS_TOTAL.endswith("_total")


class TestRecordToolUsage:
    """Test _record_tool_usage function."""

    def test_no_tools_records_nothing(self):
        request = MagicMock()
        request.tools = None

        with (
            patch.object(_agentic_calls_total, "add") as mock_agentic,
            patch.object(_tool_types_used_total, "add") as mock_types,
        ):
            _record_tool_usage(request)
            mock_agentic.assert_not_called()
            mock_types.assert_not_called()

    def test_empty_tools_records_nothing(self):
        request = MagicMock()
        request.tools = []

        with (
            patch.object(_agentic_calls_total, "add") as mock_agentic,
            patch.object(_tool_types_used_total, "add") as mock_types,
        ):
            _record_tool_usage(request)
            mock_agentic.assert_not_called()
            mock_types.assert_not_called()

    def test_function_tool_records_both_counters(self):
        tool = MagicMock()
        tool.type = "function"
        request = MagicMock()
        request.tools = [tool]

        with (
            patch.object(_agentic_calls_total, "add") as mock_agentic,
            patch.object(_tool_types_used_total, "add") as mock_types,
        ):
            _record_tool_usage(request)
            mock_agentic.assert_called_once_with(1)
            mock_types.assert_called_once_with(1, {"tool_type": "function"})

    def test_multiple_tool_types_records_each_once(self):
        func_tool = MagicMock()
        func_tool.type = "function"
        mcp_tool = MagicMock()
        mcp_tool.type = "mcp"
        request = MagicMock()
        request.tools = [func_tool, mcp_tool]

        with (
            patch.object(_agentic_calls_total, "add") as mock_agentic,
            patch.object(_tool_types_used_total, "add") as mock_types,
        ):
            _record_tool_usage(request)
            mock_agentic.assert_called_once_with(1)
            assert mock_types.call_count == 2
            tool_types_recorded = {call[0][1]["tool_type"] for call in mock_types.call_args_list}
            assert tool_types_recorded == {"function", "mcp"}

    def test_duplicate_tool_types_deduplicated(self):
        func1 = MagicMock()
        func1.type = "function"
        func2 = MagicMock()
        func2.type = "function"
        request = MagicMock()
        request.tools = [func1, func2]

        with (
            patch.object(_agentic_calls_total, "add") as mock_agentic,
            patch.object(_tool_types_used_total, "add") as mock_types,
        ):
            _record_tool_usage(request)
            mock_agentic.assert_called_once_with(1)
            mock_types.assert_called_once_with(1, {"tool_type": "function"})

    def test_web_search_variants_normalized(self):
        ws_preview = MagicMock()
        ws_preview.type = "web_search_preview"
        ws_regular = MagicMock()
        ws_regular.type = "web_search"
        request = MagicMock()
        request.tools = [ws_preview, ws_regular]

        with (
            patch.object(_agentic_calls_total, "add") as mock_agentic,
            patch.object(_tool_types_used_total, "add") as mock_types,
        ):
            _record_tool_usage(request)
            mock_agentic.assert_called_once_with(1)
            mock_types.assert_called_once_with(1, {"tool_type": "web_search"})

    def test_all_four_tool_types(self):
        tools = []
        for t in ["web_search", "file_search", "function", "mcp"]:
            tool = MagicMock()
            tool.type = t
            tools.append(tool)
        request = MagicMock()
        request.tools = tools

        with (
            patch.object(_agentic_calls_total, "add") as mock_agentic,
            patch.object(_tool_types_used_total, "add") as mock_types,
        ):
            _record_tool_usage(request)
            mock_agentic.assert_called_once_with(1)
            assert mock_types.call_count == 4
            tool_types_recorded = {call[0][1]["tool_type"] for call in mock_types.call_args_list}
            assert tool_types_recorded == {"web_search", "file_search", "function", "mcp"}
