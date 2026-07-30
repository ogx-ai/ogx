# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""
OpenTelemetry metrics for ogx tool runtime operations.

This module provides centralized metric definitions for tracking:
- Tool invocation metrics (total invocations, duration)

All metrics follow OpenTelemetry semantic conventions and use the ogx prefix
for consistent naming across the telemetry stack.
"""

from opentelemetry import metrics
from opentelemetry.metrics import Counter, Histogram

from .constants import (
    TOOL_DURATION,
    TOOL_INVOCATIONS_TOTAL,
)

# Get or create meter for ogx.tool_runtime
# This uses the global MeterProvider configured by OTEL auto-instrumentation
# or set explicitly via metrics.set_meter_provider()
meter = metrics.get_meter("ogx.tool_runtime", version="1.0.0")


# Tool invocation metrics
# These track tool usage patterns and performance

tool_invocations_total: Counter = meter.create_counter(
    name=TOOL_INVOCATIONS_TOTAL,
    description="Total number of tool invocations processed by the runtime",
    unit="1",
)

tool_duration: Histogram = meter.create_histogram(
    name=TOOL_DURATION,
    description="Duration of tool invocations from start to completion",
    unit="s",
)


# Utility function for creating metric attributes
def create_tool_metric_attributes(
    tool_group: str | None = None,
    tool_name: str | None = None,
    provider: str | None = None,
    status: str | None = None,
) -> dict[str, str]:
    """Create a consistent attribute dictionary for tool runtime metrics.

    Args:
        tool_group: Tool group ID (e.g., "websearch", "rag_tool", "custom_tools")
        tool_name: Specific tool name (e.g., "brave-search", "web_search")
        provider: Provider ID (e.g., "brave-search::impl", "file-search::impl")
        status: Request outcome ("success", "error")

    Returns:
        Dictionary of attributes with non-None values
    """
    attributes: dict[str, str] = {}

    if tool_group is not None:
        attributes["tool_group"] = tool_group
    if tool_name is not None:
        attributes["tool_name"] = tool_name
    if provider is not None:
        attributes["provider"] = provider
    if status is not None:
        attributes["status"] = status

    return attributes
