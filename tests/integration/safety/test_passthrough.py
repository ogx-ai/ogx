# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Integration tests for the passthrough safety provider.

These tests spin up a lightweight mock downstream server and wire up a
PassthroughSafetyAdapter against it, exercising the full path from
adapter through httpx to a real HTTP endpoint and back.

Run with:
    uv run pytest tests/integration/safety/test_passthrough.py -x -q
"""

import json
import threading
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from ogx.providers.remote.safety.passthrough.config import PassthroughSafetyConfig
from ogx.providers.remote.safety.passthrough.passthrough import PassthroughSafetyAdapter
from ogx_api import (
    OpenAIUserMessageParam,
    ResourceType,
    RunModerationRequest,
    RunShieldRequest,
    Shield,
)

# -- mock downstream /v1/moderations server --


class _ModerationHandler(BaseHTTPRequestHandler):
    """Handles POST /v1/moderations and records received headers."""

    # shared state set by the test fixture
    received_headers: dict[str, str] = {}
    response_override: dict[str, Any] | None = None

    def do_POST(self):  # noqa: N802 - required by BaseHTTPRequestHandler
        # record all headers the adapter sent
        _ModerationHandler.received_headers = dict(self.headers)

        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length)) if length else {}

        if self.response_override is not None:
            resp = self.response_override
        else:
            # build a default "safe" response with one result per input
            inputs = body.get("input", [])
            if isinstance(inputs, str):
                inputs = [inputs]
            results = [{"flagged": False, "categories": {}, "category_scores": {}}] * len(inputs)
            resp = {
                "id": f"modr-{uuid.uuid4()}",
                "model": body.get("model", "text-moderation-latest"),
                "results": results,
            }

        payload = json.dumps(resp).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format, *args):
        pass  # suppress server logs during tests


@pytest.fixture(scope="module")
def mock_server():
    """Start a local HTTP server that mimics /v1/moderations."""
    server = HTTPServer(("127.0.0.1", 0), _ModerationHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{port}/v1"
    server.shutdown()


async def _make_adapter(base_url: str, **kwargs) -> PassthroughSafetyAdapter:
    config = PassthroughSafetyConfig(base_url=base_url, **kwargs)
    adapter = PassthroughSafetyAdapter(config)
    await adapter.initialize()
    adapter.__provider_spec__ = MagicMock()
    adapter.__provider_spec__.provider_data_validator = (
        "ogx.providers.remote.safety.passthrough.config.PassthroughProviderDataValidator"
    )

    shield_store = AsyncMock()
    shield_store.get_shield.return_value = Shield(
        provider_id="passthrough",
        type=ResourceType.shield,
        identifier="test-shield",
        provider_resource_id="text-moderation-latest",
    )
    adapter.shield_store = shield_store
    return adapter


# -- tests --


async def test_run_shield_safe_content(mock_server):
    """run_shield returns no violation for safe content."""
    adapter = await _make_adapter(mock_server)
    request = RunShieldRequest(
        shield_id="test-shield",
        messages=[OpenAIUserMessageParam(content="hello")],
    )
    response = await adapter.run_shield(request)
    assert response.violation is None


async def test_run_moderation_safe_content(mock_server):
    """run_moderation returns unflagged result for safe content."""
    adapter = await _make_adapter(mock_server)
    request = RunModerationRequest(input="hello", model="text-moderation-latest")
    response = await adapter.run_moderation(request)
    assert len(response.results) >= 1
    assert response.results[0].flagged is False


async def test_run_shield_flagged_content(mock_server):
    """run_shield returns a violation when downstream flags content."""
    _ModerationHandler.response_override = {
        "id": "modr-flagged",
        "model": "text-moderation-latest",
        "results": [{"flagged": True, "categories": {"hate": True}, "category_scores": {"hate": 0.99}}],
    }
    try:
        adapter = await _make_adapter(mock_server)
        request = RunShieldRequest(
            shield_id="test-shield",
            messages=[OpenAIUserMessageParam(content="bad content")],
        )
        response = await adapter.run_shield(request)
        assert response.violation is not None
        assert response.violation.metadata["violation_type"] == "hate"
    finally:
        _ModerationHandler.response_override = None


async def test_api_key_sent_as_bearer(mock_server):
    """Config api_key is sent as Authorization: Bearer header."""
    adapter = await _make_adapter(mock_server, api_key="test-secret-key")
    request = RunModerationRequest(input="test", model="text-moderation-latest")
    await adapter.run_moderation(request)

    assert _ModerationHandler.received_headers.get("Authorization") == "Bearer test-secret-key"


async def test_forward_headers_sent_downstream(mock_server):
    """forward_headers mapping sends provider data as HTTP headers."""
    from ogx.core.request_headers import request_provider_data_context

    adapter = await _make_adapter(
        mock_server,
        forward_headers={"tenant_id": "X-Tenant-Id"},
    )

    provider_data = json.dumps({"tenant_id": "t-integration-test"})
    with request_provider_data_context({"x-ogx-provider-data": provider_data}):
        request = RunModerationRequest(input="test", model="text-moderation-latest")
        await adapter.run_moderation(request)

    assert _ModerationHandler.received_headers.get("X-Tenant-Id") == "t-integration-test"


async def test_sensitive_headers_rejected_at_config_time(mock_server):
    """Blocked headers raise ValidationError at config parse time, not at request time."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="blocked"):
        await _make_adapter(
            mock_server,
            forward_headers={"encoding": "Transfer-Encoding", "tenant_id": "X-Tenant-Id"},
        )


async def test_multiple_messages_all_checked(mock_server):
    """Multiple messages produce multiple results, later violations are caught."""
    _ModerationHandler.response_override = {
        "id": "modr-multi",
        "model": "text-moderation-latest",
        "results": [
            {"flagged": False, "categories": {}, "category_scores": {}},
            {"flagged": True, "categories": {"violence": True}, "category_scores": {"violence": 0.95}},
        ],
    }
    try:
        adapter = await _make_adapter(mock_server)
        request = RunShieldRequest(
            shield_id="test-shield",
            messages=[
                OpenAIUserMessageParam(content="safe message"),
                OpenAIUserMessageParam(content="unsafe message"),
            ],
        )
        response = await adapter.run_shield(request)
        assert response.violation is not None
        assert response.violation.metadata["violation_type"] == "violence"
    finally:
        _ModerationHandler.response_override = None


async def test_connection_error_wrapped(mock_server):
    """Connection to unreachable host produces RuntimeError."""
    adapter = await _make_adapter("http://127.0.0.1:1")  # port 1 -- should fail
    request = RunModerationRequest(input="test", model="text-moderation-latest")
    with pytest.raises(RuntimeError, match="Failed to reach downstream safety service"):
        await adapter.run_moderation(request)
