# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import json
import os
from typing import Any

import httpx
import pytest

from ogx.core.library_client import OGXAsLibraryClient
from ogx.core.testing_context import get_test_context

# Import fixtures from common module to make them available in this test directory
from tests.integration.fixtures.common import (  # noqa: F401
    openai_client,
    require_server,
)


def pytest_configure(config):
    """Disable stderr pipe to prevent Rich logging from blocking on buffer saturation."""
    os.environ["OGX_TEST_LOG_STDERR"] = "0"


@pytest.fixture(scope="session")
def messages_base_url(ogx_client):
    """Provide the base URL for the Messages API, skipping library client mode."""
    if isinstance(ogx_client, OGXAsLibraryClient):
        pytest.skip("Messages API tests are not supported in library client mode")
    return ogx_client.base_url


@pytest.fixture
def messages_client(messages_base_url):
    """Provide an httpx client configured for Anthropic Messages API calls."""
    client = httpx.Client(base_url=messages_base_url, timeout=60.0)
    yield client
    client.close()


def _build_messages_body(
    *,
    model: str,
    messages: list[dict],
    max_tokens: int = 256,
    stream: bool = False,
    system: str | None = None,
    tools: list[dict] | None = None,
    tool_choice: dict | str | None = None,
    temperature: float | None = None,
    stop_sequences: list[str] | None = None,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "stream": stream,
    }
    if system is not None:
        body["system"] = system
    if tools is not None:
        body["tools"] = tools
    if tool_choice is not None:
        body["tool_choice"] = tool_choice
    if temperature is not None:
        body["temperature"] = temperature
    if stop_sequences is not None:
        body["stop_sequences"] = stop_sequences
    return body


def _build_headers() -> dict[str, str]:
    headers = {
        "content-type": "application/json",
        "anthropic-version": "2023-06-01",
    }
    test_id = get_test_context()
    if test_id:
        provider_data = {"__test_id": test_id}
        headers["X-OGX-Provider-Data"] = json.dumps(provider_data)
    return headers


def make_messages_request(
    client: httpx.Client,
    **kwargs: Any,
) -> httpx.Response:
    """Make a non-streaming POST request to /v1/messages."""
    body = _build_messages_body(**kwargs)
    return client.post("/v1/messages", headers=_build_headers(), json=body)


def make_streaming_messages_request(
    client: httpx.Client,
    **kwargs: Any,
) -> list[dict]:
    """Make a streaming POST request to /v1/messages and return parsed SSE events.

    Raises AssertionError if the response status is not 200.
    """
    kwargs["stream"] = True
    body = _build_messages_body(**kwargs)
    headers = _build_headers()

    events: list[dict] = []
    current_event_type: str | None = None

    with client.stream("POST", "/v1/messages", headers=headers, json=body) as response:
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        for line in response.iter_lines():
            if line.startswith("event: "):
                current_event_type = line[7:]
            elif line.startswith("data: "):
                data = json.loads(line[6:])
                if current_event_type:
                    data["_event_type"] = current_event_type
                events.append(data)
                current_event_type = None

    return events
