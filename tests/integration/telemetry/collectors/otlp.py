# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""OTLP HTTP telemetry collector used for server-mode tests."""

import gzip
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn

from opentelemetry.proto.collector.metrics.v1.metrics_service_pb2 import ExportMetricsServiceRequest
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest

from .base import BaseTelemetryCollector, MetricStub, SpanStub, attributes_to_dict


class OtlpHttpTestCollector(BaseTelemetryCollector):
    def __init__(self) -> None:
        super().__init__()
        self._spans: list[SpanStub] = []
        self._metrics: list[MetricStub] = []
        self._lock = threading.Lock()

        class _ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
            daemon_threads = True
            allow_reuse_address = True

        configured_port = int(os.environ.get("OGX_TEST_COLLECTOR_PORT", "0"))

        self._server = _ThreadingHTTPServer(("127.0.0.1", configured_port), _CollectorHandler)
        self._server.collector = self  # type: ignore[attr-defined]
        port = self._server.server_address[1]
        self.endpoint = f"http://127.0.0.1:{port}"

        self._thread = threading.Thread(target=self._server.serve_forever, name="otel-test-collector", daemon=True)
        self._thread.start()

    def _handle_traces(self, request: ExportTraceServiceRequest) -> None:
        new_spans: list[SpanStub] = []

        for resource_spans in request.resource_spans:
            resource_attrs = attributes_to_dict(resource_spans.resource.attributes)

            for scope_spans in resource_spans.scope_spans:
                for span in scope_spans.spans:
                    new_spans.append(self._create_span_stub_from_protobuf(span, resource_attrs or None))

        if not new_spans:
            return

        with self._lock:
            self._spans.extend(new_spans)

    def _handle_metrics(self, request: ExportMetricsServiceRequest) -> None:
        new_metrics: list[MetricStub] = []
        for resource_metrics in request.resource_metrics:
            for scope_metrics in resource_metrics.scope_metrics:
                for metric in scope_metrics.metrics:
                    # Handle multiple data points per metric (e.g., different attribute sets)
                    metric_stubs = self._create_metric_stubs_from_protobuf(metric)
                    new_metrics.extend(metric_stubs)

        if not new_metrics:
            return

        with self._lock:
            self._metrics.extend(new_metrics)

    def _snapshot_spans(self) -> tuple[SpanStub, ...]:
        with self._lock:
            return tuple(self._spans)

    def _snapshot_metrics(self) -> tuple[MetricStub, ...] | None:
        with self._lock:
            return tuple(self._metrics) if self._metrics else None

    def _clear_impl(self) -> None:
        """Clear telemetry over a period of time to prevent race conditions between tests."""
        with self._lock:
            self._spans.clear()
            self._metrics.clear()

        # Prevent race conditions where telemetry arrives after clear() but before
        # the test starts, causing contamination between tests
        deadline = time.time() + 2.0  # Maximum wait time
        last_span_count = 0
        last_metric_count = 0
        stable_iterations = 0

        while time.time() < deadline:
            with self._lock:
                current_span_count = len(self._spans)
                current_metric_count = len(self._metrics)

            if current_span_count == last_span_count and current_metric_count == last_metric_count:
                stable_iterations += 1
                if stable_iterations >= 4:  # 4 * 50ms = 200ms of stability
                    break
            else:
                stable_iterations = 0
                last_span_count = current_span_count
                last_metric_count = current_metric_count

            time.sleep(0.05)

        # Final clear to remove any telemetry that arrived during stabilization
        with self._lock:
            self._spans.clear()
            self._metrics.clear()

    def shutdown(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=1)


class _CollectorHandler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:  # noqa: N802 Function name `do_POST` should be lowercase
        collector: OtlpHttpTestCollector = self.server.collector  # type: ignore[attr-defined]
        length = int(self.headers.get("content-length", "0"))
        body = self.rfile.read(length)
        if self.headers.get("content-encoding") == "gzip":
            body = gzip.decompress(body)

        if self.path == "/v1/traces":
            request = ExportTraceServiceRequest()
            request.ParseFromString(body)
            collector._handle_traces(request)
            self._respond_ok()
        elif self.path == "/v1/metrics":
            request = ExportMetricsServiceRequest()
            request.ParseFromString(body)
            collector._handle_metrics(request)
            self._respond_ok()
        else:
            self.send_response(404)
            self.end_headers()

    def _respond_ok(self) -> None:
        self.send_response(200)
        self.end_headers()
