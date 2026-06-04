# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Unit tests for the Prometheus scrape endpoint and metric exposition."""

from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from opentelemetry.exporter.prometheus import PrometheusMetricReader
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.resources import Resource
from prometheus_client import CollectorRegistry, generate_latest

from ogx.core.server.metrics import _EXCLUDED_PATHS
from ogx.telemetry import _is_prometheus_enabled
from ogx_api.inspect_api.fastapi_routes import create_router


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


@pytest.fixture
def inspect_client():
    """A TestClient over the real Inspect router, which owns the /v1/metrics route."""
    app = FastAPI()
    app.include_router(create_router(MagicMock()))
    return TestClient(app)


class TestPrometheusEnabledFlag:
    @pytest.mark.parametrize("value", ["1", "true", "TRUE", "Yes", "on"])
    def test_truthy_values(self, monkeypatch, value):
        monkeypatch.setenv("OGX_PROMETHEUS_ENABLED", value)
        assert _is_prometheus_enabled() is True

    @pytest.mark.parametrize("value", ["", "0", "false", "no", "off", "  "])
    def test_falsy_values(self, monkeypatch, value):
        monkeypatch.setenv("OGX_PROMETHEUS_ENABLED", value)
        assert _is_prometheus_enabled() is False

    def test_unset(self, monkeypatch):
        monkeypatch.delenv("OGX_PROMETHEUS_ENABLED", raising=False)
        assert _is_prometheus_enabled() is False


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


class TestMetricsEndpoint:
    def test_endpoint_serves_prometheus_when_enabled(self, monkeypatch, inspect_client):
        monkeypatch.setenv("OGX_PROMETHEUS_ENABLED", "1")

        resp = inspect_client.get("/v1/metrics")

        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/plain")

    def test_endpoint_returns_404_when_disabled(self, monkeypatch, inspect_client):
        monkeypatch.delenv("OGX_PROMETHEUS_ENABLED", raising=False)

        resp = inspect_client.get("/v1/metrics")

        assert resp.status_code == 404

    def test_endpoint_is_public(self):
        """The /v1/metrics route must opt out of auth via PUBLIC_ROUTE_KEY so collectors
        can scrape it without credentials."""
        from fastapi.routing import APIRoute

        from ogx_api.router_utils import PUBLIC_ROUTE_KEY

        router = create_router(MagicMock())
        metrics_routes = [r for r in router.routes if isinstance(r, APIRoute) and r.path.endswith("/metrics")]
        assert metrics_routes, "metrics route not registered"
        assert (metrics_routes[0].openapi_extra or {}).get(PUBLIC_ROUTE_KEY) is True


def test_metrics_path_excluded_from_request_metrics():
    """Scraping the metrics endpoint must not be counted by RequestMetricsMiddleware."""
    assert "/v1/metrics" in _EXCLUDED_PATHS
