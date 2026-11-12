# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""SSRF regression tests for caller-supplied MCP server_url in /v1/responses.

The Responses API lets callers attach an MCP server by URL. Those URLs are
untrusted input: the server must reject any endpoint that resolves to a
non-public address before opening a connection, and must never connect to
internal services. Reaching a private MCP server is supported through
admin-configured connectors (see test_tool_responses.py).

These tests need no recordings: rejection happens before the model is called.
"""

import socket
import threading

import ogx_client
import openai
import pytest

from .helpers import skip_if_provider_is_vertexai


class _ConnectionCounter:
    """A local stand-in for an internal service, counting every connection."""

    def __init__(self) -> None:
        self._server = socket.socket()
        self._server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server.bind(("127.0.0.1", 0))
        self.port = self._server.getsockname()[1]
        self._server.listen(16)
        # Poll so the accept loop notices close() promptly
        self._server.settimeout(0.2)
        self.connections = 0
        self._closed = False
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def _serve(self) -> None:
        while True:
            try:
                conn, _ = self._server.accept()
            except TimeoutError:
                if self._closed:
                    return
                continue
            except OSError:
                return
            self.connections += 1
            conn.close()

    def close(self) -> None:
        self._closed = True
        self._server.close()
        self._thread.join(timeout=5)


@pytest.fixture
def internal_service():
    service = _ConnectionCounter()
    yield service
    service.close()


def _bad_request_types():
    return (ogx_client.exceptions.BadRequestException, openai.BadRequestError)


@pytest.mark.parametrize(
    "server_url",
    [
        "http://127.0.0.1:{port}/sse",
        "http://localhost:{port}/sse",
        "http://[::1]:{port}/sse",
    ],
)
def test_mcp_ssrf_loopback_rejected_without_connecting(
    responses_client, client_with_models, text_model_id, internal_service, server_url
):
    """A caller-supplied server_url pointing at loopback must be rejected with a 400,
    and the internal service must receive zero connections."""
    skip_if_provider_is_vertexai(
        client_with_models, text_model_id, "MCP tool calling behavior differs from expected output structure"
    )
    url = server_url.format(port=internal_service.port)
    with pytest.raises(_bad_request_types()):
        responses_client.responses.create(
            model=text_model_id,
            input="What is the boiling point of myawesomeliquid in Celsius?",
            tools=[{"type": "mcp", "server_label": "internal", "server_url": url}],
            stream=False,
        )
    assert internal_service.connections == 0


@pytest.mark.parametrize(
    "server_url",
    [
        "http://10.0.0.1/sse",  # RFC1918
        "http://169.254.169.254/latest/meta-data/",  # cloud metadata endpoint
        "http://100.64.0.1/sse",  # CGNAT
        "http://[fd00::1]/sse",  # IPv6 unique-local
        "file:///etc/passwd",  # non-http scheme
    ],
)
def test_mcp_ssrf_non_public_endpoints_rejected(responses_client, client_with_models, text_model_id, server_url):
    """Non-public IP literals and non-http schemes must be rejected with a 400."""
    skip_if_provider_is_vertexai(
        client_with_models, text_model_id, "MCP tool calling behavior differs from expected output structure"
    )
    with pytest.raises(_bad_request_types()):
        responses_client.responses.create(
            model=text_model_id,
            input="What is the boiling point of myawesomeliquid in Celsius?",
            tools=[{"type": "mcp", "server_label": "internal", "server_url": server_url}],
            stream=False,
        )
