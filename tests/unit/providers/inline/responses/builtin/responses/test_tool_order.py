# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""The chat tools sent to the model keep the request's tool order, whether an
MCP listing is reused from the previous response or listed again."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from ogx.providers.inline.responses.builtin.responses.streaming import StreamingResponseOrchestrator
from ogx.providers.inline.responses.builtin.responses.types import ChatCompletionContext, ToolContext
from ogx_api import (
    OpenAIResponseInputToolFileSearch,
    OpenAIResponseInputToolMCP,
    OpenAIResponseObject,
    OpenAIResponseToolMCP,
    ToolDef,
)

TOOLS = [
    OpenAIResponseInputToolFileSearch(vector_store_ids=["fake"]),
    OpenAIResponseInputToolMCP(server_label="alabel", server_url="http://localhost:9999/mcp"),
]


async def _chat_tool_names(tool_context: ToolContext) -> tuple[list[str], list]:
    ctx = MagicMock(spec=ChatCompletionContext)
    ctx.tool_context = tool_context
    ctx.chat_tools = None
    orchestrator = StreamingResponseOrchestrator(
        inference_api=AsyncMock(),
        ctx=ctx,
        response_id="resp_test",
        created_at=0,
        text=MagicMock(),
        max_infer_iters=1,
        tool_executor=MagicMock(),
        instructions=None,
        moderation_endpoint=None,
    )
    output: list = []
    mcp_tools = SimpleNamespace(data=[ToolDef(name="mcp_tool", description="d", input_schema={})])
    with patch(
        "ogx.providers.inline.responses.builtin.responses.streaming.list_mcp_tools",
        AsyncMock(return_value=mcp_tools),
    ):
        async for _ in orchestrator._process_tools(output):
            pass
    return [t["function"]["name"] for t in ctx.chat_tools], output


async def test_reused_mcp_listing_keeps_request_tool_order():
    first_turn, output = await _chat_tool_names(ToolContext(TOOLS))
    assert first_turn == ["file_search", "mcp_tool"]

    previous = OpenAIResponseObject(created_at=1, id="prev", model="m", output=output, status="", store=True)
    previous.tools = [TOOLS[0], OpenAIResponseToolMCP(server_label="alabel")]
    second_context = ToolContext(TOOLS)
    second_context.recover_tools_from_previous_response(previous)
    assert len(second_context.previous_tool_listings) == 1

    second_turn, _ = await _chat_tool_names(second_context)
    assert second_turn == first_turn
