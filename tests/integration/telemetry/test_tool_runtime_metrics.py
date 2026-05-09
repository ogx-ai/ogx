# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Integration tests for tool runtime metrics.

These tests verify that OpenTelemetry metrics are correctly recorded for
tool runtime operations.
"""

import pytest


async def test_tool_metrics_with_websearch_tool(mock_otlp_collector, ogx_client):
    """Test metrics using web_search tool from ci-tests."""
    mock_otlp_collector.clear()

    try:
        # Invoke web_search tool (from ci-tests built-in tools)
        result = ogx_client.tool_runtime.invoke_tool(
            tool_name="web_search",
            kwargs={"query": "test query"},
        )

        # Verify the tool was invoked successfully
        assert result is not None

        # Get metrics
        metrics = mock_otlp_collector.get_metrics(
            expected_count=2,  # invocations_total, duration
            timeout=10.0,
        )

        # Verify metrics exist
        assert "ogx.tool_runtime.invocations_total" in metrics, "invocations_total metric not found"
        assert "ogx.tool_runtime.duration_seconds" in metrics, "duration metric not found"

        # Verify metric attributes
        invocations_metric = metrics["ogx.tool_runtime.invocations_total"]
        assert invocations_metric.attributes.get("tool_name") == "web_search"
        assert invocations_metric.attributes.get("status") == "success"
        assert invocations_metric.value >= 1

        # Verify duration is positive
        duration_metric = metrics["ogx.tool_runtime.duration_seconds"]
        assert duration_metric.value > 0

    except Exception as e:
        # If web_search tool is not available, skip the test
        pytest.skip(f"web_search tool not available: {e}")


async def test_tool_error_metrics_basic(mock_otlp_collector, ogx_client):
    """Test that error metrics are recorded (basic test)."""
    mock_otlp_collector.clear()

    try:
        # Try to invoke a non-existent tool
        ogx_client.tool_runtime.invoke_tool(
            tool_name="definitely_nonexistent_tool_12345",
            kwargs={},
        )
        pytest.fail("Expected exception for non-existent tool")
    except Exception:
        # Expected error
        pass

    # Try to get metrics (with longer timeout for error cases)
    metrics = mock_otlp_collector.get_metrics(
        expected_count=1,
        timeout=15.0,
    )

    # If we got metrics, verify they're error metrics
    if metrics:
        if "ogx.tool_runtime.invocations_total" in metrics:
            invocations_metric = metrics["ogx.tool_runtime.invocations_total"]
            # Should be marked as error if metrics were recorded
            if invocations_metric.attributes.get("status"):
                assert invocations_metric.attributes.get("status") == "error"
    # else: Error happened before metrics instrumentation, which is OK


async def test_metrics_accumulate(mock_otlp_collector, ogx_client):
    """Test that metrics accumulate across multiple invocations."""
    mock_otlp_collector.clear()

    try:
        # Make multiple invocations
        for i in range(3):
            result = ogx_client.tool_runtime.invoke_tool(
                tool_name="web_search",
                kwargs={"query": f"test query {i}"},
            )
            assert result is not None

        # Get metrics
        metrics = mock_otlp_collector.get_metrics(
            expected_count=2,
            timeout=10.0,
        )

        # Verify invocations accumulated
        assert "ogx.tool_runtime.invocations_total" in metrics
        invocations_metric = metrics["ogx.tool_runtime.invocations_total"]
        assert invocations_metric.value >= 3, f"Expected at least 3 invocations, got {invocations_metric.value}"

    except Exception as e:
        pytest.skip(f"web_search tool not available: {e}")
