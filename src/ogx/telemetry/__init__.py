# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""OpenTelemetry initialization for ogx.

This module configures OpenTelemetry metrics export based on environment variables.
If OTEL_EXPORTER_OTLP_ENDPOINT is set, metrics will be exported to that endpoint.
"""

import os

from ogx.log import get_logger

logger = get_logger(__name__, category="telemetry")


def setup_telemetry() -> None:
    """Initialize OpenTelemetry metrics exporter if configured via environment.

    This function checks for OTEL_EXPORTER_OTLP_ENDPOINT and configures the
    MeterProvider to export metrics to the specified endpoint.
    """
    otlp_endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")

    if not otlp_endpoint:
        logger.debug("OTEL_EXPORTER_OTLP_ENDPOINT not set, metrics will not be exported")
        return

    try:
        from opentelemetry import metrics
        from opentelemetry.exporter.otlp.proto.http.metric_exporter import (
            OTLPMetricExporter,
        )
        from opentelemetry.sdk.metrics import MeterProvider
        from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
        from opentelemetry.sdk.resources import Resource

        # Get export interval from environment (default 200ms for tests, 60s otherwise)
        export_interval_ms = int(os.environ.get("OTEL_METRIC_EXPORT_INTERVAL", "60000"))
        export_interval_s = export_interval_ms / 1000.0

        # Create OTLP exporter
        exporter = OTLPMetricExporter(endpoint=f"{otlp_endpoint}/v1/metrics")

        # Create metric reader with periodic export
        reader = PeriodicExportingMetricReader(exporter, export_interval_millis=export_interval_ms)

        # Create resource with service name
        service_name = os.environ.get("OTEL_SERVICE_NAME", "ogx")
        resource = Resource(attributes={"service.name": service_name})

        # Create and set global MeterProvider
        provider = MeterProvider(resource=resource, metric_readers=[reader])
        metrics.set_meter_provider(provider)

        logger.info(
            "OpenTelemetry metrics exporter configured: (interval: s)",
            otlp_endpoint=otlp_endpoint,
            export_interval_s=export_interval_s,
        )

    except Exception as e:
        logger.warning("Failed to configure OpenTelemetry metrics exporter", error=str(e))


# Initialize telemetry when module is imported
setup_telemetry()
