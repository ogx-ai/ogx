# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Telemetry tests verifying @trace_protocol decorator format across stack modes.

Note: The mock_otlp_collector fixture automatically clears telemetry data
before and after each test, ensuring test isolation.
"""

import json

import pytest


def test_streaming_chunk_count(mock_otlp_collector, ogx_client, text_model_id):
    """Verify streaming adds chunk_count and __type__=async_generator."""

    pytest.skip("Disabled: See https://github.com/ogx-ai/ogx/issues/4089")
    stream = ogx_client.chat.completions.create(
        model=text_model_id,
        messages=[{"role": "user", "content": "Test trace openai 1"}],
        stream=True,
    )

    chunks = list(stream)
    assert len(chunks) > 0

    spans = mock_otlp_collector.get_spans(expected_count=5)
    assert len(spans) > 0

    async_generator_span = next(
        (
            span
            for span in reversed(spans)
            if span.get_span_type() == "async_generator"
            and span.attributes.get("chunk_count")
            and span.has_message("Test trace openai 1")
        ),
        None,
    )

    assert async_generator_span is not None

    raw_chunk_count = async_generator_span.attributes.get("chunk_count")
    assert raw_chunk_count is not None
    chunk_count = int(raw_chunk_count)

    assert chunk_count == len(chunks)


def test_telemetry_format_completeness(mock_otlp_collector, ogx_client, text_model_id):
    """Comprehensive validation of telemetry data format including spans and metrics."""

    pytest.skip("Disabled: See https://github.com/ogx-ai/ogx/issues/4089")
    response = ogx_client.chat.completions.create(
        model=text_model_id,
        messages=[{"role": "user", "content": "Test trace openai with temperature 0.7"}],
        temperature=0.7,
        max_tokens=100,
        stream=False,
    )

    # Handle both dict and Pydantic model for usage
    # This occurs due to the replay system returning a dict for usage, but the client returning a Pydantic model
    # TODO: Fix this by making the replay system return a Pydantic model for usage
    usage = response.usage if isinstance(response.usage, dict) else response.usage.model_dump()
    assert usage.get("prompt_tokens") and usage["prompt_tokens"] > 0
    assert usage.get("completion_tokens") and usage["completion_tokens"] > 0
    assert usage.get("total_tokens") and usage["total_tokens"] > 0

    # Verify spans
    spans = mock_otlp_collector.get_spans(expected_count=7)
    target_span = next(
        (span for span in reversed(spans) if span.has_message("Test trace openai with temperature 0.7")),
        None,
    )
    assert target_span is not None

    trace_id = target_span.get_trace_id()
    assert trace_id is not None

    spans = [span for span in spans if span.get_trace_id() == trace_id]
    spans = [span for span in spans if span.is_root_span() or span.is_autotraced()]
    assert len(spans) >= 4

    # Collect all model_ids found in spans
    logged_model_ids = []

    for span in spans:
        attrs = span.attributes
        assert attrs is not None

        # Root span is created manually by tracing middleware, not by @trace_protocol decorator
        if span.is_root_span():
            assert span.get_location() in ["library_client", "server"]
            continue

        assert span.is_autotraced()
        class_name, method_name = span.get_class_method()
        assert class_name and method_name
        assert span.get_span_type() in ["async", "sync", "async_generator"]

        args_field = span.attributes.get("__args__")
        if args_field:
            args = json.loads(args_field)
            if "model_id" in args:
                logged_model_ids.append(args["model_id"])

    # At least one span should capture the fully qualified model ID
    assert text_model_id in logged_model_ids, f"Expected to find {text_model_id} in spans, but got {logged_model_ids}"

    # Verify token usage metrics in response using polling
    expected_metrics = ["completion_tokens", "total_tokens", "prompt_tokens"]
    metrics = mock_otlp_collector.get_metrics(expected_count=len(expected_metrics), expect_model_id=text_model_id)
    assert len(metrics) > 0, "No metrics found within timeout"

    # Filter metrics to only those from the specific model used in the request
    # Multiple metrics with the same name can exist (e.g., from safety models)
    inference_model_metrics = {}
    all_model_ids = set()

    for name, metric in metrics.items():
        if name in expected_metrics:
            model_id = metric.attributes.get("model_id")
            all_model_ids.add(model_id)
            # Only include metrics from the specific model used in the test request
            if model_id == text_model_id:
                inference_model_metrics[name] = metric

    # Verify expected metrics are present for our specific model
    for metric_name in expected_metrics:
        assert metric_name in inference_model_metrics, (
            f"Expected metric {metric_name} for model {text_model_id} not found. "
            f"Available models: {sorted(all_model_ids)}, "
            f"Available metrics for {text_model_id}: {list(inference_model_metrics.keys())}"
        )

    # Verify metric values match usage data
    assert inference_model_metrics["completion_tokens"].value == usage["completion_tokens"], (
        f"Expected {usage['completion_tokens']} for completion_tokens, but got {inference_model_metrics['completion_tokens'].value}"
    )
    assert inference_model_metrics["total_tokens"].value == usage["total_tokens"], (
        f"Expected {usage['total_tokens']} for total_tokens, but got {inference_model_metrics['total_tokens'].value}"
    )
    assert inference_model_metrics["prompt_tokens"].value == usage["prompt_tokens"], (
        f"Expected {usage['prompt_tokens']} for prompt_tokens, but got {inference_model_metrics['prompt_tokens'].value}"
    )
