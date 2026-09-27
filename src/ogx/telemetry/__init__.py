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

initialize_telemetry() is the entry point: it configures the metric readers and starts the
scrape server. It is called from Stack.initialize() (server and library modes), not at
import, so commands that merely import this module (e.g. `ogx stack list-deps`) neither
configure telemetry nor open a network port.

When the process is launched with `opentelemetry-instrument`, the auto-instrumentation owns
the global MeterProvider and manages OTLP export. In that case setup_telemetry() adds only
the Prometheus scrape reader to the existing provider rather than installing a competing one.
"""

import os

from ogx.log import get_logger

logger = get_logger(__name__, category="telemetry")

# Default port for the metrics scrape server, matching the OpenTelemetry Prometheus convention.
_DEFAULT_METRICS_PORT = 9464

# Guards initialize_telemetry() so repeated stack initializations in one process (e.g. in
# tests) don't reconfigure telemetry, which would duplicate the Prometheus collector or
# rebind the scrape port.
_telemetry_initialized = False


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


def initialize_telemetry() -> None:
    """Configure metric export and start the scrape server.

    Single entry point invoked once from a serving path (Stack.initialize()). Guarded so
    repeated stack initializations in one process do not reconfigure telemetry or rebind the
    scrape port.
    """
    global _telemetry_initialized
    if _telemetry_initialized:
        return
    _telemetry_initialized = True

    setup_telemetry()
    start_metrics_server()


def setup_telemetry() -> None:
    """Configure OpenTelemetry metric export based on environment configuration.

    Adds an OTLP push reader when OTEL_EXPORTER_OTLP_ENDPOINT is set and a Prometheus scrape
    reader when OGX_METRICS_ENDPOINT_ENABLED is truthy.

    If an SDK MeterProvider is already installed — e.g. when the process is launched with
    `opentelemetry-instrument`, which owns OTLP export — the Prometheus scrape reader is
    added to that provider instead of installing a competing one. Installing a second
    provider is rejected by OpenTelemetry ("Overriding of current MeterProvider is not
    allowed") and would leave ogx metrics off the scrape endpoint. Otherwise a new
    MeterProvider is created and installed.
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

        existing_provider = metrics.get_meter_provider()
        if isinstance(existing_provider, MeterProvider):
            # A provider is already installed (e.g. by opentelemetry-instrument, which manages
            # OTLP export). Add only the Prometheus scrape reader — ogx's addition — to it, so
            # ogx metrics reach the scrape endpoint without displacing the existing provider.
            if metrics_endpoint_enabled:
                from opentelemetry.exporter.prometheus import PrometheusMetricReader

                existing_provider.add_metric_reader(PrometheusMetricReader())
                logger.info("Added Prometheus scrape reader to the existing MeterProvider")
            return

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

            # Registers a collector on the default prometheus_client registry; the HTTP server
            # that serves it is started by start_metrics_server().
            metric_readers.append(PrometheusMetricReader())
            logger.info("OpenTelemetry metrics scrape reader configured")

        service_name = os.environ.get("OTEL_SERVICE_NAME", "ogx")
        attributes: dict[str, str] = {"service.name": service_name}

        namespace = os.environ.get("OTEL_SERVICE_NAMESPACE", "").strip() or os.environ.get("NAMESPACE", "").strip()
        if namespace:
            attributes["service.namespace"] = namespace

        cluster_id = os.environ.get("CLUSTER_ID", "").strip()
        if cluster_id:
            attributes["k8s.cluster.uid"] = cluster_id

        resource = Resource(attributes=attributes)

        provider = MeterProvider(resource=resource, metric_readers=metric_readers)
        metrics.set_meter_provider(provider)

    except Exception as e:
        logger.warning("Failed to configure OpenTelemetry metrics exporter", error=str(e))


def start_metrics_server() -> None:
    """Start the standalone metrics scrape HTTP server when the endpoint is enabled.

    Serves the default prometheus_client registry that setup_telemetry()'s
    PrometheusMetricReader writes to. Raises if OGX_METRICS_PORT is misconfigured, failing
    startup fast.
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
