# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Regression tests for ApprovalFilter in _approval_required().

Verifies that the ApprovalFilter.always and ApprovalFilter.never lists
are correctly evaluated when require_approval is an ApprovalFilter object.

See: https://github.com/ogx-ai/ogx/issues/5287
"""

from unittest.mock import AsyncMock, MagicMock

from ogx.providers.inline.responses.builtin.responses.streaming import StreamingResponseOrchestrator
from ogx.providers.inline.responses.builtin.responses.types import ChatCompletionContext, ToolContext
from ogx_api.openai_responses import (
    ApprovalFilter,
    OpenAIResponseInputToolMCP,
)


def _build_orchestrator(mcp_tool_to_server: dict[str, OpenAIResponseInputToolMCP]) -> StreamingResponseOrchestrator:
    """Build a minimal orchestrator with the given MCP tool mapping."""
    mock_ctx = MagicMock(spec=ChatCompletionContext)
    mock_ctx.tool_context = MagicMock(spec=ToolContext)
    mock_ctx.tool_context.previous_tools = mcp_tool_to_server
    mock_ctx.model = "test-model"
    mock_ctx.messages = []
    mock_ctx.temperature = None
    mock_ctx.top_p = None
    mock_ctx.frequency_penalty = None
    mock_ctx.response_format = MagicMock()
    mock_ctx.tool_choice = None

    orchestrator = StreamingResponseOrchestrator(
        inference_api=AsyncMock(),
        ctx=mock_ctx,
        response_id="resp_test",
        created_at=0,
        text=MagicMock(),
        max_infer_iters=1,
        tool_executor=MagicMock(),
        instructions=None,
        safety_api=None,
    )
    return orchestrator


def _make_mcp_server(**kwargs) -> OpenAIResponseInputToolMCP:
    defaults = {"server_label": "test-server", "server_url": "http://localhost:9999/mcp"}
    defaults.update(kwargs)
    return OpenAIResponseInputToolMCP(**defaults)


class TestApprovalFilterCorrectness:
    """Verify ApprovalFilter.always and .never lists are respected."""

    def test_never_list_skips_approval(self):
        mcp_server = _make_mcp_server(
            require_approval=ApprovalFilter(never=["safe_tool"]),
        )
        orch = _build_orchestrator({"safe_tool": mcp_server})
        assert orch._approval_required("safe_tool") is False

    def test_always_list_requires_approval(self):
        mcp_server = _make_mcp_server(
            require_approval=ApprovalFilter(always=["dangerous_tool"]),
        )
        orch = _build_orchestrator({"dangerous_tool": mcp_server})
        assert orch._approval_required("dangerous_tool") is True

    def test_unlisted_tool_defaults_to_approval(self):
        mcp_server = _make_mcp_server(
            require_approval=ApprovalFilter(always=["other"], never=["other2"]),
        )
        orch = _build_orchestrator({"unlisted_tool": mcp_server})
        assert orch._approval_required("unlisted_tool") is True

    def test_unknown_tool_returns_false(self):
        orch = _build_orchestrator({})
        assert orch._approval_required("nonexistent") is False

    def test_literal_always_requires_approval(self):
        mcp_server = _make_mcp_server(require_approval="always")
        orch = _build_orchestrator({"any_tool": mcp_server})
        assert orch._approval_required("any_tool") is True

    def test_literal_never_skips_approval(self):
        mcp_server = _make_mcp_server(require_approval="never")
        orch = _build_orchestrator({"any_tool": mcp_server})
        assert orch._approval_required("any_tool") is False

    def test_filter_with_both_lists(self):
        mcp_server = _make_mcp_server(
            require_approval=ApprovalFilter(
                always=["dangerous_tool"],
                never=["safe_tool"],
            ),
        )
        mapping = {
            "safe_tool": mcp_server,
            "dangerous_tool": mcp_server,
            "unlisted_tool": mcp_server,
        }
        orch = _build_orchestrator(mapping)

        assert orch._approval_required("safe_tool") is False
        assert orch._approval_required("dangerous_tool") is True
        assert orch._approval_required("unlisted_tool") is True
