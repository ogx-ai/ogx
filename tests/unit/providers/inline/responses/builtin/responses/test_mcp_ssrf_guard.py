# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Regression tests for the MCP tool ``server_url`` SSRF guard.

An unauthenticated caller could point an MCP tool's ``server_url`` at an internal
address (loopback, RFC1918, link-local, or cloud metadata) and have the server
connect to it and forward attacker-supplied headers/tokens. The server must reject
such URLs before opening a connection.

See: https://github.com/ogx-ai/ogx/issues/6287
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ogx.providers.inline.responses.builtin.responses import streaming
from ogx.providers.inline.responses.builtin.responses.streaming import StreamingResponseOrchestrator
from ogx.providers.inline.responses.builtin.responses.types import ChatCompletionContext, ToolContext
from ogx_api import OpenAIResponseInputToolMCP


def _make_mcp_server(**kwargs) -> OpenAIResponseInputToolMCP:
    defaults = {"server_label": "test-server", "server_url": "http://localhost:9999/mcp"}
    defaults.update(kwargs)
    return OpenAIResponseInputToolMCP(**defaults)


def _build_orchestrator() -> StreamingResponseOrchestrator:
    mock_ctx = MagicMock(spec=ChatCompletionContext)
    mock_ctx.tool_context = MagicMock(spec=ToolContext)
    mock_ctx.tool_context.previous_tools = {}
    mock_ctx.model = "test-model"
    mock_ctx.messages = []
    mock_ctx.temperature = None
    mock_ctx.top_p = None
    mock_ctx.frequency_penalty = None
    mock_ctx.response_format = MagicMock()
    mock_ctx.tool_choice = None
    mock_ctx.response_tools = []
    mock_ctx.approval_response = MagicMock(return_value=None)

    return StreamingResponseOrchestrator(
        inference_api=AsyncMock(),
        ctx=mock_ctx,
        response_id="resp_test",
        created_at=0,
        text=MagicMock(),
        max_infer_iters=1,
        tool_executor=MagicMock(),
        instructions=None,
        moderation_endpoint=None,
    )


class TestMcpServerUrlSsrfGuard:
    """MCP tool ``server_url`` must be validated to prevent SSRF (issue #6287)."""

    async def test_private_server_url_rejected_before_connection(self):
        orch = _build_orchestrator()
        mcp_tool = _make_mcp_server(server_url="http://169.254.169.254/latest/meta-data/")

        with patch.object(streaming, "list_mcp_tools", new_callable=AsyncMock) as mock_list:
            with pytest.raises(ValueError, match="private"):
                async for _ in orch._process_mcp_tool(mcp_tool, ["seed"]):
                    pass

        # The connection must never be opened for a blocked URL.
        mock_list.assert_not_called()

    async def test_loopback_server_url_rejected(self):
        orch = _build_orchestrator()
        mcp_tool = _make_mcp_server(server_url="http://127.0.0.1:19877/mcp")

        with patch.object(streaming, "list_mcp_tools", new_callable=AsyncMock) as mock_list:
            with pytest.raises(ValueError, match="private"):
                async for _ in orch._process_mcp_tool(mcp_tool, ["seed"]):
                    pass

        mock_list.assert_not_called()

    async def test_public_server_url_still_allowed(self):
        orch = _build_orchestrator()
        mcp_tool = _make_mcp_server(server_url="http://8.8.8.8/mcp")

        with patch.object(streaming, "list_mcp_tools", new_callable=AsyncMock) as mock_list:
            async for _ in orch._process_mcp_tool(mcp_tool, ["seed"]):
                pass

        mock_list.assert_awaited_once()
        assert mock_list.await_args.kwargs["endpoint"] == "http://8.8.8.8/mcp"
