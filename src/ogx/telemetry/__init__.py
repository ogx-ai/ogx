# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""OpenTelemetry initialization for ogx.

This module configures OpenTelemetry metrics export based on environment variables.
Two export paths can be enabled independently and simultaneously:
- OTLP push: enabled when OTEL_EXPORTER_OTLP_ENDPOINT is set.
- Prometheus scrape: enabled when OGX_PROMETHEUS_ENABLED is truthy, exposing metrics
  at the server's /metrics endpoint in Prometheus exposition format.
"""

import os

from ogx.log import get_logger

logger = get_logger(__name__, category="telemetry")


def _is_prometheus_enabled() -> bool:
    """Return True if the Prometheus scrape endpoint is enabled via environment."""
    return os.environ.get("OGX_PROMETHEUS_ENABLED", "").strip().lower() in ("1", "true", "yes", "on")


def setup_telemetry() -> None:
    """Initialize OpenTelemetry metric readers based on environment configuration.

    Adds an OTLP push reader when OTEL_EXPORTER_OTLP_ENDPOINT is set and a Prometheus
    scrape reader when OGX_PROMETHEUS_ENABLED is truthy. Both readers attach to a single
    global MeterProvider, so the two export paths operate independently. If neither is
    configured, no MeterProvider is installed and metrics are not exported.
    """
    otlp_endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
    prometheus_enabled = _is_prometheus_enabled()

    if not otlp_endpoint and not prometheus_enabled:
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

        if prometheus_enabled:
            from opentelemetry.exporter.prometheus import PrometheusMetricReader

            # Registers a collector on the default prometheus_client registry; the server's
            # /metrics endpoint serves it via prometheus_client.generate_latest().
            metric_readers.append(PrometheusMetricReader())
            logger.info("OpenTelemetry Prometheus metrics reader configured, metrics exposed at /metrics")

        service_name = os.environ.get("OTEL_SERVICE_NAME", "ogx")
        resource = Resource(attributes={"service.name": service_name})

        provider = MeterProvider(resource=resource, metric_readers=metric_readers)
        metrics.set_meter_provider(provider)

    except Exception as e:
        logger.warning("Failed to configure OpenTelemetry metrics exporter", error=str(e))


# Initialize telemetry when module is imported
setup_telemetry()
