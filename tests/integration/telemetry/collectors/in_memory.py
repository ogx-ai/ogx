# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""In-memory telemetry collector for library-client tests."""

import opentelemetry.metrics as otel_metrics
import opentelemetry.trace as otel_trace
from opentelemetry import metrics, trace
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from .base import BaseTelemetryCollector, MetricStub, SpanStub


# TODO: Fix thi to work with Automatic Instrumentation
class InMemoryTelemetryCollector(BaseTelemetryCollector):
    """In-memory telemetry collector for library-client tests.

    Converts OpenTelemetry span objects to SpanStub objects to ensure
    consistent interface with OTLP collector used in server mode.
    """

    def __init__(self, span_exporter: InMemorySpanExporter, metric_reader: InMemoryMetricReader) -> None:
        super().__init__()
        self._span_exporter = span_exporter
        self._metric_reader = metric_reader

    def _snapshot_spans(self) -> tuple[SpanStub, ...]:
        spans = []
        for span in self._span_exporter.get_finished_spans():
            spans.append(self._create_span_stub_from_opentelemetry(span))
        return tuple(spans)

    def _snapshot_metrics(self) -> tuple[MetricStub, ...] | None:
        data = self._metric_reader.get_metrics_data()
        if not data or not data.resource_metrics:
            return None

        metric_stubs = []
        for resource_metric in data.resource_metrics:
            if resource_metric.scope_metrics:
                for scope_metric in resource_metric.scope_metrics:
                    for metric in scope_metric.metrics:
                        metric_stub = self._extract_metric_from_opentelemetry(metric)
                        if metric_stub:
                            metric_stubs.append(metric_stub)

        return tuple(metric_stubs) if metric_stubs else None

    def _clear_impl(self) -> None:
        self._span_exporter.clear()
        self._metric_reader.get_metrics_data()


class InMemoryTelemetryManager:
    def __init__(self) -> None:
        if hasattr(otel_trace, "_TRACER_PROVIDER_SET_ONCE"):
            otel_trace._TRACER_PROVIDER_SET_ONCE._done = False  # type: ignore[attr-defined]
        if hasattr(otel_metrics, "_METER_PROVIDER_SET_ONCE"):
            otel_metrics._METER_PROVIDER_SET_ONCE._done = False  # type: ignore[attr-defined]

        span_exporter = InMemorySpanExporter()
        tracer_provider = TracerProvider()
        tracer_provider.add_span_processor(SimpleSpanProcessor(span_exporter))
        trace.set_tracer_provider(tracer_provider)

        metric_reader = InMemoryMetricReader()
        meter_provider = MeterProvider(metric_readers=[metric_reader])
        metrics.set_meter_provider(meter_provider)

        self.collector = InMemoryTelemetryCollector(span_exporter, metric_reader)
        self._tracer_provider = tracer_provider
        self._meter_provider = meter_provider

    def shutdown(self) -> None:
        self._tracer_provider.shutdown()
        self._meter_provider.shutdown()
