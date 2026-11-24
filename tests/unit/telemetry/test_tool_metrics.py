# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Unit tests for tool metrics."""

from ogx.telemetry.tool_metrics import (
    tool_duration,
    tool_invocations_total,
)


class TestToolMetricInstruments:
    """Test that metric instruments are properly defined."""

    def test_tool_invocations_total_exists(self):
        """Test that tool_invocations_total counter exists."""
        assert tool_invocations_total is not None
        assert hasattr(tool_invocations_total, "add")

    def test_tool_duration_exists(self):
        """Test that tool_duration histogram exists."""
        assert tool_duration is not None
        assert hasattr(tool_duration, "record")

    def test_tool_invocations_total_can_record(self):
        """Test that we can record metrics without errors."""
        # This should not raise any exceptions
        tool_invocations_total.add(
            1,
            {"tool_name": "web_search", "provider": "brave-search::impl", "status": "success"},
        )

    def test_tool_duration_can_record(self):
        """Test that we can record duration metrics without errors."""
        # This should not raise any exceptions
        tool_duration.record(1.234, {"tool_name": "web_search", "provider": "brave-search::impl"})


class TestToolMetricsConstants:
    """Test that metric constants are properly defined."""

    def test_metric_names_follow_convention(self):
        """Test that metric names follow OpenTelemetry naming conventions."""
        from ogx.telemetry.constants import (
            TOOL_DURATION,
            TOOL_INVOCATIONS_TOTAL,
        )

        # Should start with the ogx.tools prefix
        assert TOOL_INVOCATIONS_TOTAL.startswith("ogx.tools.")
        assert TOOL_DURATION.startswith("ogx.tools.")

        # Should follow semantic naming
        assert TOOL_INVOCATIONS_TOTAL.endswith("_total")
        assert TOOL_DURATION.endswith("_seconds")
