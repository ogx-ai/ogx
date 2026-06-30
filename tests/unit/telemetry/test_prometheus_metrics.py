# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Unit tests for the Prometheus scrape server and metric exposition."""

import pytest
from opentelemetry.exporter.prometheus import PrometheusMetricReader
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.resources import Resource
from prometheus_client import CollectorRegistry, generate_latest

from ogx.telemetry import _DEFAULT_METRICS_PORT, _is_metrics_endpoint_enabled, _metrics_port


@pytest.fixture
def prometheus_meter_provider():
    """A MeterProvider wired to a PrometheusMetricReader backed by an isolated registry.

    Using a dedicated CollectorRegistry keeps the test independent of the process-global
    prometheus_client registry and of OGX's global MeterProvider.
    """
    registry = CollectorRegistry()
    reader = PrometheusMetricReader(registry=registry)
    provider = MeterProvider(
        resource=Resource(attributes={"service.name": "ogx-test"}),
        metric_readers=[reader],
    )
    yield provider, registry
    provider.shutdown()


class TestMetricsEndpointEnabledFlag:
    @pytest.mark.parametrize("value", ["1", "true", "TRUE", "Yes", "on"])
    def test_truthy_values(self, monkeypatch, value):
        monkeypatch.setenv("OGX_METRICS_ENDPOINT_ENABLED", value)
        assert _is_metrics_endpoint_enabled() is True

    @pytest.mark.parametrize("value", ["", "0", "false", "no", "off", "  "])
    def test_falsy_values(self, monkeypatch, value):
        monkeypatch.setenv("OGX_METRICS_ENDPOINT_ENABLED", value)
        assert _is_metrics_endpoint_enabled() is False

    def test_unset(self, monkeypatch):
        monkeypatch.delenv("OGX_METRICS_ENDPOINT_ENABLED", raising=False)
        assert _is_metrics_endpoint_enabled() is False


class TestMetricsPort:
    def test_default_when_unset(self, monkeypatch):
        monkeypatch.delenv("OGX_METRICS_PORT", raising=False)
        assert _metrics_port() == _DEFAULT_METRICS_PORT

    def test_override(self, monkeypatch):
        monkeypatch.setenv("OGX_METRICS_PORT", "9999")
        assert _metrics_port() == 9999

    def test_invalid_raises(self, monkeypatch):
        """A misconfigured port must fail fast rather than silently use the default."""
        monkeypatch.setenv("OGX_METRICS_PORT", "not-a-port")
        with pytest.raises(ValueError, match="OGX_METRICS_PORT"):
            _metrics_port()


class TestPrometheusExposition:
    def test_metrics_exposed_in_prometheus_format(self, prometheus_meter_provider):
        provider, registry = prometheus_meter_provider

        meter = provider.get_meter("ogx.test")
        counter = meter.create_counter(name="ogx_test_requests_total", unit="1")
        counter.add(3, {"api": "models", "status": "success"})

        output = generate_latest(registry).decode("utf-8")

        # Prometheus exposition format: the counter surfaces with a _total suffix,
        # carries its labels, and exposes the recorded value.
        assert "ogx_test_requests_total" in output
        assert 'api="models"' in output
        assert 'status="success"' in output
        assert "3.0" in output
