# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Shared helpers for telemetry test collectors."""

import os
import time
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any


@dataclass
class MetricStub:
    """Unified metric interface for both in-memory and OTLP collectors."""

    name: str
    value: Any
    attributes: dict[str, Any] | None = None


@dataclass
class SpanStub:
    """Unified span interface for both in-memory and OTLP collectors."""

    name: str
    attributes: dict[str, Any] | None = None
    resource_attributes: dict[str, Any] | None = None
    events: list[dict[str, Any]] | None = None
    trace_id: str | None = None
    span_id: str | None = None

    @property
    def context(self):
        """Provide context-like interface for trace_id compatibility."""
        if self.trace_id is None:
            return None
        return type("Context", (), {"trace_id": int(self.trace_id, 16)})()

    def get_trace_id(self) -> str | None:
        """Get trace ID in hex format.

        Tries context.trace_id first, then falls back to direct trace_id.
        """
        context = getattr(self, "context", None)
        if context and getattr(context, "trace_id", None) is not None:
            return f"{context.trace_id:032x}"
        return getattr(self, "trace_id", None)

    def has_message(self, text: str) -> bool:
        """Check if span contains a specific message in its args."""
        if self.attributes is None:
            return False
        args = self.attributes.get("__args__")
        if not args or not isinstance(args, str):
            return False
        return text in args

    def is_root_span(self) -> bool:
        """Check if this is a root span."""
        if self.attributes is None:
            return False
        return self.attributes.get("__root__") is True

    def is_autotraced(self) -> bool:
        """Check if this span was automatically traced."""
        if self.attributes is None:
            return False
        return self.attributes.get("__autotraced__") is True

    def get_span_type(self) -> str | None:
        """Get the span type (async, sync, async_generator)."""
        if self.attributes is None:
            return None
        return self.attributes.get("__type__")

    def get_class_method(self) -> tuple[str | None, str | None]:
        """Get the class and method names for autotraced spans."""
        if self.attributes is None:
            return None, None
        return (self.attributes.get("__class__"), self.attributes.get("__method__"))

    def get_location(self) -> str | None:
        """Get the location (library_client, server) for root spans."""
        if self.attributes is None:
            return None
        return self.attributes.get("__location__")


def _value_to_python(value: Any) -> Any:
    kind = value.WhichOneof("value")
    if kind == "string_value":
        return value.string_value
    if kind == "int_value":
        return value.int_value
    if kind == "double_value":
        return value.double_value
    if kind == "bool_value":
        return value.bool_value
    if kind == "bytes_value":
        return value.bytes_value
    if kind == "array_value":
        return [_value_to_python(item) for item in value.array_value.values]
    if kind == "kvlist_value":
        return {kv.key: _value_to_python(kv.value) for kv in value.kvlist_value.values}
    return None


def attributes_to_dict(key_values: Iterable[Any]) -> dict[str, Any]:
    return {key_value.key: _value_to_python(key_value.value) for key_value in key_values}


def events_to_list(events: Iterable[Any]) -> list[dict[str, Any]]:
    return [
        {
            "name": event.name,
            "timestamp": event.time_unix_nano,
            "attributes": attributes_to_dict(event.attributes),
        }
        for event in events
    ]


class BaseTelemetryCollector:
    """Base class for telemetry collectors that ensures consistent return types.

    All collectors must return SpanStub objects to ensure test compatibility
    across both library-client and server modes.
    """

    # Default delay in seconds if OTEL_METRIC_EXPORT_INTERVAL is not set
    _DEFAULT_BASELINE_STABILIZATION_DELAY = 0.2

    def __init__(self):
        self._metric_baseline: dict[tuple[str, str], float] = {}

    @classmethod
    def _get_baseline_stabilization_delay(cls) -> float:
        """Get baseline stabilization delay from OTEL_METRIC_EXPORT_INTERVAL.

        Adds 1.5x buffer for CI environments.
        """
        interval_ms = os.environ.get("OTEL_METRIC_EXPORT_INTERVAL")
        if interval_ms:
            try:
                delay = float(interval_ms) / 1000.0
            except (ValueError, TypeError):
                delay = cls._DEFAULT_BASELINE_STABILIZATION_DELAY
        else:
            delay = cls._DEFAULT_BASELINE_STABILIZATION_DELAY

        if os.environ.get("CI"):
            delay *= 1.5

        return delay

    def _get_metric_key(self, metric: MetricStub) -> tuple[str, str]:
        """Generate a stable key for a metric based on name and attributes."""
        attrs = metric.attributes or {}
        attr_key = ",".join(f"{k}={v}" for k, v in sorted(attrs.items()))
        return (metric.name, attr_key)

    def _compute_metric_delta(self, metric: MetricStub) -> int | float | None:
        """Compute delta value for a metric from baseline.

        Returns:
            Delta value if metric was in baseline, absolute value if new, None if unchanged.
        """
        metric_key = self._get_metric_key(metric)

        if metric_key in self._metric_baseline:
            baseline_value = self._metric_baseline[metric_key]
            delta = metric.value - baseline_value
            return delta if delta > 0 else None
        else:
            return metric.value

    def get_spans(
        self,
        expected_count: int | None = None,
        timeout: float = 5.0,
        poll_interval: float = 0.05,
    ) -> tuple[SpanStub, ...]:
        deadline = time.time() + timeout
        min_count = expected_count if expected_count is not None else 1
        last_len: int | None = None
        stable_iterations = 0

        while True:
            spans = tuple(self._snapshot_spans())

            if len(spans) >= min_count:
                if expected_count is not None and len(spans) >= expected_count:
                    return spans

                if last_len == len(spans):
                    stable_iterations += 1
                    if stable_iterations >= 2:
                        return spans
                else:
                    stable_iterations = 1
            else:
                stable_iterations = 0

            if time.time() >= deadline:
                return spans

            last_len = len(spans)
            time.sleep(poll_interval)

    def get_metrics(
        self,
        expected_count: int | None = None,
        timeout: float = 5.0,
        poll_interval: float = 0.05,
        expect_model_id: str | None = None,
    ) -> dict[str, MetricStub]:
        """Poll until expected metrics are available or timeout is reached.

        Returns metrics with delta values computed from baseline.
        """
        deadline = time.time() + timeout
        min_count = expected_count if expected_count is not None else 1
        accumulated_metrics = {}
        seen_metric_names_with_model_id = set()

        while time.time() < deadline:
            current_metrics = self._snapshot_metrics()
            if current_metrics:
                for metric in current_metrics:
                    delta_value = self._compute_metric_delta(metric)
                    if delta_value is None:
                        continue

                    metric_with_delta = MetricStub(
                        name=metric.name,
                        value=delta_value,
                        attributes=metric.attributes,
                    )

                    self._accumulate_metric(
                        accumulated_metrics,
                        metric_with_delta,
                        expect_model_id,
                        seen_metric_names_with_model_id,
                    )

            if self._has_enough_metrics(
                accumulated_metrics, seen_metric_names_with_model_id, min_count, expect_model_id
            ):
                return accumulated_metrics

            time.sleep(poll_interval)

        return accumulated_metrics

    def _accumulate_metric(
        self,
        accumulated: dict[str, MetricStub],
        metric: MetricStub,
        expect_model_id: str | None,
        seen_with_model_id: set[str],
    ) -> None:
        """Accumulate a metric, preferring those matching expected model_id."""
        metric_name = metric.name
        matches_model_id = (
            expect_model_id and metric.attributes and metric.attributes.get("model_id") == expect_model_id
        )

        if metric_name not in accumulated:
            accumulated[metric_name] = metric
            if matches_model_id:
                seen_with_model_id.add(metric_name)
            return

        existing = accumulated[metric_name]
        existing_matches = (
            expect_model_id and existing.attributes and existing.attributes.get("model_id") == expect_model_id
        )

        if matches_model_id and not existing_matches:
            accumulated[metric_name] = metric
            seen_with_model_id.add(metric_name)
        elif matches_model_id == existing_matches:
            if metric.value > existing.value:
                accumulated[metric_name] = metric
            if matches_model_id:
                seen_with_model_id.add(metric_name)

    def _has_enough_metrics(
        self,
        accumulated: dict[str, MetricStub],
        seen_with_model_id: set[str],
        min_count: int,
        expect_model_id: str | None,
    ) -> bool:
        """Check if we have collected enough metrics."""
        if len(accumulated) < min_count:
            return False
        if not expect_model_id:
            return True
        return len(seen_with_model_id) >= min_count

    @staticmethod
    def _convert_attributes_to_dict(attrs: Any) -> dict[str, Any]:
        """Convert various attribute types to a consistent dictionary format.

        Handles mappingproxy, dict, and other attribute types.
        """
        if attrs is None:
            return {}

        try:
            return dict(attrs.items())  # type: ignore[attr-defined]
        except AttributeError:
            try:
                return dict(attrs)
            except TypeError:
                return dict(attrs) if attrs else {}

    @staticmethod
    def _extract_trace_span_ids(span: Any) -> tuple[str | None, str | None]:
        """Extract trace_id and span_id from OpenTelemetry span object.

        Handles both context-based and direct attribute access.
        """
        trace_id = None
        span_id = None

        context = getattr(span, "context", None)
        if context:
            trace_id = f"{context.trace_id:032x}"
            span_id = f"{context.span_id:016x}"
        else:
            trace_id = getattr(span, "trace_id", None)
            span_id = getattr(span, "span_id", None)

        return trace_id, span_id

    @staticmethod
    def _create_span_stub_from_opentelemetry(span: Any) -> SpanStub:
        """Create SpanStub from OpenTelemetry span object.

        This helper reduces code duplication between collectors.
        """
        trace_id, span_id = BaseTelemetryCollector._extract_trace_span_ids(span)
        attributes = BaseTelemetryCollector._convert_attributes_to_dict(span.attributes) or {}

        return SpanStub(
            name=span.name,
            attributes=attributes,
            trace_id=trace_id,
            span_id=span_id,
        )

    @staticmethod
    def _create_span_stub_from_protobuf(span: Any, resource_attrs: dict[str, Any] | None = None) -> SpanStub:
        """Create SpanStub from protobuf span object.

        This helper handles the different structure of protobuf spans.
        """
        attributes = attributes_to_dict(span.attributes) or {}
        events = events_to_list(span.events) if span.events else None
        trace_id = span.trace_id.hex() if span.trace_id else None
        span_id = span.span_id.hex() if span.span_id else None

        return SpanStub(
            name=span.name,
            attributes=attributes,
            resource_attributes=resource_attrs,
            events=events,
            trace_id=trace_id,
            span_id=span_id,
        )

    @staticmethod
    def _extract_metric_from_opentelemetry(metric: Any) -> MetricStub | None:
        """Extract MetricStub from OpenTelemetry metric object.

        This helper reduces code duplication between collectors.
        """
        if not (hasattr(metric, "name") and hasattr(metric, "data") and hasattr(metric.data, "data_points")):
            return None

        if not (metric.data.data_points and len(metric.data.data_points) > 0):
            return None

        data_point = metric.data.data_points[0]

        if hasattr(data_point, "value"):
            # Counter or Gauge
            value = data_point.value
        elif hasattr(data_point, "sum"):
            # Histogram - use the sum of all recorded values
            value = data_point.sum
        else:
            return None

        attributes = {}
        if hasattr(data_point, "attributes"):
            attrs = data_point.attributes
            if attrs is not None and hasattr(attrs, "items"):
                attributes = dict(attrs.items())
            elif attrs is not None and not isinstance(attrs, dict):
                attributes = dict(attrs)

        return MetricStub(
            name=metric.name,
            value=value,
            attributes=attributes or {},
        )

    @staticmethod
    def _create_metric_stubs_from_protobuf(metric: Any) -> list[MetricStub]:
        """Create list of MetricStub objects from protobuf metric object.

        Protobuf metrics can have sum, gauge, or histogram data. Each metric can have
        multiple data points with different attributes, so we return one MetricStub
        per data point.

        Returns:
            List of MetricStub objects, one per data point in the metric.
        """
        if not hasattr(metric, "name"):
            return []

        metric_stubs = []

        for metric_type in ["sum", "gauge", "histogram"]:
            if not hasattr(metric, metric_type):
                continue

            metric_data = getattr(metric, metric_type)
            if not metric_data or not hasattr(metric_data, "data_points"):
                continue

            data_points = metric_data.data_points
            if not data_points:
                continue

            for data_point in data_points:
                attributes = attributes_to_dict(data_point.attributes) if hasattr(data_point, "attributes") else {}

                value = BaseTelemetryCollector._extract_data_point_value(data_point, metric_type)
                if value is None:
                    continue

                metric_stubs.append(
                    MetricStub(
                        name=metric.name,
                        value=value,
                        attributes=attributes,
                    )
                )

            # Only process one metric type per metric
            break

        return metric_stubs

    @staticmethod
    def _extract_data_point_value(data_point: Any, metric_type: str) -> float | int | None:
        """Extract value from a protobuf metric data point based on metric type."""
        if metric_type == "sum":
            if hasattr(data_point, "as_int"):
                return data_point.as_int
            if hasattr(data_point, "as_double"):
                return data_point.as_double
        elif metric_type == "gauge":
            if hasattr(data_point, "as_double"):
                return data_point.as_double
        elif metric_type == "histogram":
            # Histograms use sum field which represents cumulative sum of all recorded values
            if hasattr(data_point, "sum"):
                return data_point.sum

        return None

    def clear(self) -> None:
        """Clear telemetry data and establish baseline for metric delta computation."""
        self._metric_baseline.clear()

        self._clear_impl()

        delay = self._get_baseline_stabilization_delay()
        time.sleep(delay)
        baseline_metrics = self._snapshot_metrics()
        if baseline_metrics:
            for metric in baseline_metrics:
                metric_key = self._get_metric_key(metric)
                self._metric_baseline[metric_key] = metric.value

    def _snapshot_spans(self) -> tuple[SpanStub, ...]:  # pragma: no cover - interface hook
        raise NotImplementedError

    def _snapshot_metrics(self) -> tuple[MetricStub, ...] | None:  # pragma: no cover - interface hook
        raise NotImplementedError

    def _clear_impl(self) -> None:  # pragma: no cover - interface hook
        raise NotImplementedError

    def shutdown(self) -> None:
        """Optional hook for subclasses with background workers."""
