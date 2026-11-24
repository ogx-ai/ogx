# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""
OpenTelemetry metrics for ogx tool operations.

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

# Get or create meter for ogx.tools
# This uses the global MeterProvider configured by OTEL auto-instrumentation
# or set explicitly via metrics.set_meter_provider()
meter = metrics.get_meter("ogx.tools", version="1.0.0")


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
