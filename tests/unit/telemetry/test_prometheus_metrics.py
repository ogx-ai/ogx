# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Unit tests for the Prometheus scrape server and metric exposition."""

import pytest
from opentelemetry import metrics as otel_metrics
from opentelemetry.exporter.prometheus import PrometheusMetricReader
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.resources import Resource
from prometheus_client import CollectorRegistry, generate_latest

import ogx.telemetry as telemetry
from ogx.telemetry import _DEFAULT_METRICS_PORT, _is_metrics_endpoint_enabled, _metrics_port, setup_telemetry


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


class TestSetupTelemetryProvider:
    """setup_telemetry() must cooperate with a pre-existing MeterProvider.

    When ogx is launched under `opentelemetry-instrument`, the auto-instrumentation installs
    the global MeterProvider. setup_telemetry() must attach the scrape reader to it rather
    than installing a competing provider (which OpenTelemetry rejects, leaving ogx metrics
    off the scrape endpoint). Real PrometheusMetricReader is swapped for InMemoryMetricReader
    to keep the process-global prometheus_client registry untouched.
    """

    def test_adds_reader_to_existing_provider(self, monkeypatch):
        monkeypatch.setenv("OGX_METRICS_ENDPOINT_ENABLED", "1")
        monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
        monkeypatch.setattr("opentelemetry.exporter.prometheus.PrometheusMetricReader", InMemoryMetricReader)

        existing = MeterProvider()  # stands in for the opentelemetry-instrument provider
        added: list = []
        set_calls: list = []
        monkeypatch.setattr(existing, "add_metric_reader", added.append)
        monkeypatch.setattr(otel_metrics, "get_meter_provider", lambda: existing)
        monkeypatch.setattr(otel_metrics, "set_meter_provider", set_calls.append)

        setup_telemetry()

        # A reader was attached to the existing provider, and it was not overridden.
        assert len(added) == 1
        assert isinstance(added[0], InMemoryMetricReader)
        assert set_calls == []
        existing.shutdown()

    def test_creates_provider_when_none_exists(self, monkeypatch):
        monkeypatch.setenv("OGX_METRICS_ENDPOINT_ENABLED", "1")
        monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
        monkeypatch.setattr("opentelemetry.exporter.prometheus.PrometheusMetricReader", InMemoryMetricReader)

        # Before set_meter_provider(), OTel returns a proxy, which is not an SDK MeterProvider.
        set_calls: list = []
        monkeypatch.setattr(otel_metrics, "get_meter_provider", object)
        monkeypatch.setattr(otel_metrics, "set_meter_provider", set_calls.append)

        setup_telemetry()

        assert len(set_calls) == 1
        assert isinstance(set_calls[0], MeterProvider)
        set_calls[0].shutdown()


def test_initialize_telemetry_is_idempotent(monkeypatch):
    """Repeated stack initializations in one process must configure telemetry only once."""
    monkeypatch.setattr(telemetry, "_telemetry_initialized", False)
    calls = {"setup": 0, "server": 0}
    monkeypatch.setattr(telemetry, "setup_telemetry", lambda: calls.__setitem__("setup", calls["setup"] + 1))
    monkeypatch.setattr(telemetry, "start_metrics_server", lambda: calls.__setitem__("server", calls["server"] + 1))

    telemetry.initialize_telemetry()
    telemetry.initialize_telemetry()
    telemetry.initialize_telemetry()

    assert calls == {"setup": 1, "server": 1}


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
