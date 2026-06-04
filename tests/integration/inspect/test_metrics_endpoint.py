# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Integration tests for the Prometheus /v1/metrics scrape endpoint.

These only run in server mode with the Prometheus reader enabled: the endpoint is
exposed over HTTP only when the server is started with OGX_PROMETHEUS_ENABLED, which
scripts/integration-tests.sh sets for native server-mode runs. The endpoint is scraped
with a raw HTTP client (not the typed SDK) because it returns Prometheus text rather
than JSON and is intentionally absent from the OpenAPI spec.
"""

import os

import httpx
import pytest

_PROMETHEUS_ENABLED = os.environ.get("OGX_PROMETHEUS_ENABLED", "").strip().lower() in ("1", "true", "yes", "on")

pytestmark = pytest.mark.skipif(
    os.environ.get("OGX_TEST_STACK_CONFIG_TYPE") != "server" or not _PROMETHEUS_ENABLED,
    reason="The Prometheus /v1/metrics endpoint is only exposed by the HTTP server when OGX_PROMETHEUS_ENABLED is set",
)


def _server_base_url(ogx_client) -> str:
    """Root server URL (without the /v1 suffix) for raw HTTP scrapes."""
    return str(ogx_client.base_url).rstrip("/").removesuffix("/v1")


def test_metrics_endpoint_exposes_prometheus_format(ogx_client):
    base_url = _server_base_url(ogx_client)

    # Exercise a non-excluded endpoint so request-level metrics are recorded.
    for _ in range(3):
        httpx.get(f"{base_url}/v1/health", timeout=30.0)

    resp = httpx.get(f"{base_url}/v1/metrics", timeout=30.0)

    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/plain")

    body = resp.text
    # Prometheus exposition format markers.
    assert "# HELP" in body
    assert "# TYPE" in body
    # OGX request metrics recorded by RequestMetricsMiddleware, proving OTel metrics
    # flow through the PrometheusMetricReader to the scrape endpoint.
    assert "ogx_requests_total" in body
    assert 'method="health"' in body


def test_metrics_endpoint_requires_no_auth(ogx_client):
    """The scrape endpoint must be reachable without an Authorization header."""
    base_url = _server_base_url(ogx_client)

    resp = httpx.get(f"{base_url}/v1/metrics", timeout=30.0)

    assert resp.status_code == 200


def test_metrics_endpoint_is_not_self_counted(ogx_client):
    """Scraping /v1/metrics must not be counted by RequestMetricsMiddleware."""
    base_url = _server_base_url(ogx_client)

    httpx.get(f"{base_url}/v1/metrics", timeout=30.0)
    body = httpx.get(f"{base_url}/v1/metrics", timeout=30.0).text

    assert 'method="metrics"' not in body
