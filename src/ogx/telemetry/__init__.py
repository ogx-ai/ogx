# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""OpenTelemetry initialization for ogx.

This module configures OpenTelemetry metrics export based on environment variables.
Two export paths can be enabled independently and simultaneously:
- OTLP push: enabled when OTEL_EXPORTER_OTLP_ENDPOINT is set.
- Metrics scrape endpoint: enabled when OGX_METRICS_ENDPOINT_ENABLED is truthy, exposing
  metrics in Prometheus exposition format on a dedicated HTTP server (OGX_METRICS_PORT,
  default 9464; OGX_METRICS_HOST, default 127.0.0.1). The scrape server listens on its own
  port, separate from the main API, so that metrics can be collected without API
  authentication and without being reachable by regular API consumers. It binds to loopback
  by default; set OGX_METRICS_HOST to expose it to other hosts or pods.

setup_telemetry() configures the metric readers at import time, while start_metrics_server()
binds the scrape server's port. The bind is deferred to the server run path so commands that
merely import this module (e.g. `ogx stack list-deps`) do not open a network port.
"""

import os

from ogx.log import get_logger

logger = get_logger(__name__, category="telemetry")

# Default port for the metrics scrape server, matching the OpenTelemetry Prometheus convention.
_DEFAULT_METRICS_PORT = 9464


def _is_metrics_endpoint_enabled() -> bool:
    """Return True if the standalone metrics scrape endpoint is enabled via environment."""
    return os.environ.get("OGX_METRICS_ENDPOINT_ENABLED", "").strip().lower() in ("1", "true", "yes", "on")


def _metrics_port() -> int:
    """Return the port for the metrics scrape server (OGX_METRICS_PORT, default 9464).

    Raises ValueError when OGX_METRICS_PORT is set to a non-integer, so a misconfiguration
    fails fast at startup rather than silently serving on an unexpected port.
    """
    raw = os.environ.get("OGX_METRICS_PORT", "").strip()
    if not raw:
        return _DEFAULT_METRICS_PORT
    try:
        return int(raw)
    except ValueError as e:
        raise ValueError(f"Failed to parse OGX_METRICS_PORT as an integer: {raw!r}") from e


def setup_telemetry() -> None:
    """Initialize OpenTelemetry metric readers based on environment configuration.

    Adds an OTLP push reader when OTEL_EXPORTER_OTLP_ENDPOINT is set and a Prometheus
    scrape reader when OGX_METRICS_ENDPOINT_ENABLED is truthy. Both readers attach to a
    single global MeterProvider, so the two export paths operate independently. The scrape
    reader only registers a collector here; the HTTP server that serves it is started
    separately by start_metrics_server(). If neither path is configured, no MeterProvider
    is installed and metrics are not exported.
    """
    otlp_endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
    metrics_endpoint_enabled = _is_metrics_endpoint_enabled()

    if not otlp_endpoint and not metrics_endpoint_enabled:
        logger.debug("No metrics exporter configured, metrics will not be exported")
        return

    try:
        from opentelemetry import metrics
        from opentelemetry.sdk.metrics import MeterProvider
        from opentelemetry.sdk.metrics.export import MetricReader
        from opentelemetry.sdk.resources import Resource

        metric_readers: list[MetricReader] = []

        if otlp_endpoint:
            from opentelemetry.exporter.otlp.proto.http.metric_exporter import (
                OTLPMetricExporter,
            )
            from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader

            # Get export interval from environment (default 200ms for tests, 60s otherwise)
            export_interval_ms = int(os.environ.get("OTEL_METRIC_EXPORT_INTERVAL", "60000"))

            exporter = OTLPMetricExporter(endpoint=f"{otlp_endpoint}/v1/metrics")
            metric_readers.append(PeriodicExportingMetricReader(exporter, export_interval_millis=export_interval_ms))
            logger.info(
                "OpenTelemetry OTLP metrics exporter configured",
                otlp_endpoint=otlp_endpoint,
                export_interval_s=export_interval_ms / 1000.0,
            )

        if metrics_endpoint_enabled:
            from opentelemetry.exporter.prometheus import PrometheusMetricReader

            # Registers a collector on the default prometheus_client registry; the HTTP
            # server that serves it is started later by start_metrics_server() from the
            # server run path, so non-serving commands don't bind a port.
            metric_readers.append(PrometheusMetricReader())
            logger.info("OpenTelemetry metrics scrape reader configured")

        service_name = os.environ.get("OTEL_SERVICE_NAME", "ogx")
        resource = Resource(attributes={"service.name": service_name})

        provider = MeterProvider(resource=resource, metric_readers=metric_readers)
        metrics.set_meter_provider(provider)

    except Exception as e:
        logger.warning("Failed to configure OpenTelemetry metrics exporter", error=str(e))


def start_metrics_server() -> None:
    """Start the standalone metrics scrape HTTP server when the endpoint is enabled.

    Called from the server run path rather than at import time, so commands that merely
    import this module (e.g. `ogx stack list-deps`) do not bind a network port. Serves the
    default prometheus_client registry that setup_telemetry()'s PrometheusMetricReader
    writes to. Raises if OGX_METRICS_PORT is misconfigured, failing startup fast.
    """
    if not _is_metrics_endpoint_enabled():
        return

    from prometheus_client import start_http_server

    port = _metrics_port()
    # Default to loopback so metrics are not exposed off-host unless explicitly opted in;
    # set OGX_METRICS_HOST (e.g. 0.0.0.0) to expose the endpoint to other hosts or pods.
    host = os.environ.get("OGX_METRICS_HOST", "127.0.0.1").strip() or "127.0.0.1"
    start_http_server(port=port, addr=host)
    logger.info("Metrics scrape endpoint exposed", host=host, port=port)


# Initialize telemetry metric readers when module is imported. The scrape server's port is
# bound separately via start_metrics_server() from the server run path.
setup_telemetry()
