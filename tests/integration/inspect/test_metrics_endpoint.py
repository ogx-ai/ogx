# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Integration tests for the standalone metrics scrape server.

These only run in server mode with the metrics endpoint enabled: the scrape server is
started on its own port (OGX_METRICS_PORT, default 9464) only when the server is launched
with OGX_METRICS_ENDPOINT_ENABLED, which scripts/integration-tests.sh sets for native
server-mode runs. The endpoint is scraped with a raw HTTP client (not the typed SDK)
because it returns Prometheus text and lives on a dedicated port separate from the API.
"""

import os
from urllib.parse import urlsplit

import httpx
import pytest

_METRICS_ENABLED = os.environ.get("OGX_METRICS_ENDPOINT_ENABLED", "").strip().lower() in ("1", "true", "yes", "on")
_METRICS_PORT = int(os.environ.get("OGX_METRICS_PORT", "9464"))

pytestmark = pytest.mark.skipif(
    os.environ.get("OGX_TEST_STACK_CONFIG_TYPE") != "server" or not _METRICS_ENABLED,
    reason="The metrics scrape server is only started when OGX_METRICS_ENDPOINT_ENABLED is set in server mode",
)


def _api_base_url(ogx_client) -> str:
    """Root API server URL (without the /v1 suffix) for exercising regular endpoints."""
    return str(ogx_client.base_url).rstrip("/").removesuffix("/v1")


def _metrics_url(ogx_client) -> str:
    """URL of the standalone metrics scrape server, on its own port."""
    host = urlsplit(str(ogx_client.base_url)).hostname or "localhost"
    return f"http://{host}:{_METRICS_PORT}/metrics"


def test_metrics_endpoint_exposes_prometheus_format(ogx_client):
    api_base_url = _api_base_url(ogx_client)

    # Exercise a regular API endpoint so request-level metrics are recorded.
    for _ in range(3):
        httpx.get(f"{api_base_url}/v1/health", timeout=30.0)

    resp = httpx.get(_metrics_url(ogx_client), timeout=30.0)

    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/plain")

    body = resp.text
    # Prometheus exposition format markers.
    assert "# HELP" in body
    assert "# TYPE" in body
    # OGX request metrics recorded by RequestMetricsMiddleware, proving OTel metrics
    # flow through the PrometheusMetricReader to the scrape server.
    assert "ogx_requests_total" in body
    assert 'method="health"' in body


def test_metrics_endpoint_requires_no_auth(ogx_client):
    """The scrape server is on a separate port and must be reachable without auth."""
    resp = httpx.get(_metrics_url(ogx_client), timeout=30.0)

    assert resp.status_code == 200


def test_metrics_endpoint_is_off_the_api_port(ogx_client):
    """The API port must not serve a /v1/metrics route; metrics live on their own port."""
    api_base_url = _api_base_url(ogx_client)

    resp = httpx.get(f"{api_base_url}/v1/metrics", timeout=30.0)

    assert resp.status_code == 404
