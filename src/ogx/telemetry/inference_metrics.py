# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""
OpenTelemetry metrics for ogx inference operations.

This module provides centralized metric definitions for tracking:
- Inference duration (end-to-end latency for chat completions)
- Time to first token (streaming requests only)
- Tokens per second (output throughput)

All metrics follow OpenTelemetry semantic conventions and use the ogx prefix
for consistent naming across the telemetry stack.
"""

from opentelemetry import metrics
from opentelemetry.metrics import Histogram

from .constants import (
    INFERENCE_DURATION,
    INFERENCE_TIME_TO_FIRST_TOKEN,
    INFERENCE_TOKENS_PER_SECOND,
)

# Get or create meter for ogx.inference
meter = metrics.get_meter("ogx.inference", version="1.0.0")

inference_duration: Histogram = meter.create_histogram(
    name=INFERENCE_DURATION,
    description="Duration of inference requests from start to completion",
    unit="s",
)

inference_time_to_first_token: Histogram = meter.create_histogram(
    name=INFERENCE_TIME_TO_FIRST_TOKEN,
    description="Time from request start to first content token (streaming only)",
    unit="s",
)

inference_tokens_per_second: Histogram = meter.create_histogram(
    name=INFERENCE_TOKENS_PER_SECOND,
    description="Output token throughput (completion tokens / duration)",
)


def create_inference_metric_attributes(
    model: str | None = None,
    provider: str | None = None,
    stream: bool | None = None,
    status: str | None = None,
) -> dict[str, str]:
    """Create a consistent attribute dictionary for inference metrics.

    Args:
        model: Fully qualified model ID (e.g., "openai/gpt-4o-mini")
        provider: Provider ID (e.g., "openai")
        stream: Whether this is a streaming request
        status: Request outcome ("success", "error")

    Returns:
        Dictionary of attributes with non-None values
    """
    attributes: dict[str, str] = {}

    if model is not None:
        attributes["model"] = model
    if provider is not None:
        attributes["provider"] = provider
    if stream is not None:
        attributes["stream"] = str(stream).lower()
    if status is not None:
        attributes["status"] = status

    return attributes
