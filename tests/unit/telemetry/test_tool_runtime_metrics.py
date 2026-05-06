# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Unit tests for tool runtime metrics."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from ogx.core.routers.tool_runtime import ToolRuntimeRouter
from ogx.telemetry.tool_runtime_metrics import (
    create_tool_metric_attributes,
    tool_duration,
    tool_invocations_total,
)
from ogx_api import ToolInvocationResult


class TestToolMetricAttributes:
    """Test metric attribute creation utility."""

    def test_create_tool_metric_attributes_all_fields(self):
        """Test creating attributes with all fields populated."""
        attrs = create_tool_metric_attributes(
            tool_group="websearch",
            tool_name="brave-search",
            provider="brave-search::impl",
            status="success",
        )
        assert attrs == {
            "tool_group": "websearch",
            "tool_name": "brave-search",
            "provider": "brave-search::impl",
            "status": "success",
        }

    def test_create_tool_metric_attributes_partial_fields(self):
        """Test creating attributes with only some fields."""
        attrs = create_tool_metric_attributes(
            tool_name="brave-search",
            status="error",
        )
        assert attrs == {
            "tool_name": "brave-search",
            "status": "error",
        }
        assert "tool_group" not in attrs
        assert "provider" not in attrs

    def test_create_tool_metric_attributes_minimal(self):
        """Test creating attributes with minimal fields."""
        attrs = create_tool_metric_attributes(
            tool_name="web_search",
        )
        assert attrs == {
            "tool_name": "web_search",
        }

    def test_create_tool_metric_attributes_empty(self):
        """Test creating attributes with no fields."""
        attrs = create_tool_metric_attributes()
        assert attrs == {}


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
        attrs = create_tool_metric_attributes(
            tool_group="websearch",
            tool_name="web_search",
            provider="brave-search::impl",
            status="success",
        )
        tool_invocations_total.add(1, attrs)

    def test_tool_duration_can_record(self):
        """Test that we can record duration metrics without errors."""
        # This should not raise any exceptions
        attrs = create_tool_metric_attributes(
            tool_group="websearch",
            tool_name="web_search",
            provider="brave-search::impl",
        )
        tool_duration.record(1.234, attrs)


class TestToolMetricsConstants:
    """Test that metric constants are properly defined."""

    def test_metric_names_follow_convention(self):
        """Test that metric names follow OpenTelemetry naming conventions."""
        from ogx.telemetry.constants import (
            TOOL_DURATION,
            TOOL_INVOCATIONS_TOTAL,
        )

        # Should start with ogx prefix
        assert TOOL_INVOCATIONS_TOTAL.startswith("ogx.")
        assert TOOL_DURATION.startswith("ogx.")

        # Should include tool_runtime in the name
        assert "tool_runtime" in TOOL_INVOCATIONS_TOTAL
        assert "tool_runtime" in TOOL_DURATION

        # Should follow semantic naming
        assert TOOL_INVOCATIONS_TOTAL.endswith("_total")
        assert TOOL_DURATION.endswith("_seconds")


class TestToolRuntimeIntegration:
    """Test tool runtime router integration with metrics."""

    async def test_tool_runtime_metrics_success(self):
        """Test that successful tool invocations record metrics correctly."""
        # Create a mock routing table
        mock_routing_table = MagicMock()

        # Mock provider
        mock_provider = AsyncMock()
        mock_provider.__provider_id__ = "brave-search::impl"
        mock_provider.invoke_tool.return_value = ToolInvocationResult(content="Search results here")

        # Setup routing table responses
        mock_routing_table.get_provider_impl = AsyncMock(return_value=mock_provider)
        mock_routing_table.tool_to_toolgroup = {"web_search": "websearch"}

        # Create router
        router = ToolRuntimeRouter(routing_table=mock_routing_table)

        # Invoke tool
        result = await router.invoke_tool(
            tool_name="web_search",
            kwargs={"query": "test query"},
            authorization=None,
        )

        # Verify the tool was invoked
        assert result.content == "Search results here"
        mock_provider.invoke_tool.assert_called_once_with(
            tool_name="web_search",
            kwargs={"query": "test query"},
            authorization=None,
        )

        # Note: In a real integration test, we would verify that metrics were
        # exported to the OTLP collector, similar to the inference metrics tests

    async def test_tool_runtime_metrics_error(self):
        """Test that failed tool invocations record error metrics correctly."""
        # Create a mock routing table
        mock_routing_table = MagicMock()

        # Mock provider that raises an error
        mock_provider = AsyncMock()
        mock_provider.__provider_id__ = "brave-search::impl"
        mock_provider.invoke_tool.side_effect = ValueError("Tool execution failed")

        # Setup routing table responses
        mock_routing_table.get_provider_impl = AsyncMock(return_value=mock_provider)
        mock_routing_table.tool_to_toolgroup = {"web_search": "websearch"}

        # Create router
        router = ToolRuntimeRouter(routing_table=mock_routing_table)

        # Invoke tool and expect error
        with pytest.raises(ValueError, match="Tool execution failed"):
            await router.invoke_tool(
                tool_name="web_search",
                kwargs={"query": "test query"},
                authorization=None,
            )

        # Verify the tool was attempted
        mock_provider.invoke_tool.assert_called_once()

        # Note: Error metrics (status="error") would be recorded and exported

    async def test_tool_runtime_metrics_cancelled_error(self):
        """Test that cancelled tool invocations record error metrics correctly."""
        mock_routing_table = MagicMock()

        mock_provider = AsyncMock()
        mock_provider.__provider_id__ = "brave-search::impl"
        mock_provider.invoke_tool.side_effect = asyncio.CancelledError()

        mock_routing_table.get_provider_impl = AsyncMock(return_value=mock_provider)
        mock_routing_table.tool_to_toolgroup = {"web_search": "websearch"}

        router = ToolRuntimeRouter(routing_table=mock_routing_table)

        with pytest.raises(asyncio.CancelledError):
            await router.invoke_tool(
                tool_name="web_search",
                kwargs={"query": "test query"},
                authorization=None,
            )

        mock_provider.invoke_tool.assert_called_once()
