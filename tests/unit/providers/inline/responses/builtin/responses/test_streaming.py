# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from collections.abc import AsyncIterator
from unittest.mock import AsyncMock, MagicMock

import pytest

from ogx.providers.inline.responses.builtin.responses.streaming import (
    StreamingResponseOrchestrator,
    convert_tooldef_to_chat_tool,
)
from ogx.providers.inline.responses.builtin.responses.types import ChatCompletionContext, ToolContext
from ogx.providers.inline.responses.builtin.responses.utils import (
    build_summary_prompt,
    should_summarize_reasoning,
    summarize_reasoning,
)
from ogx_api import ToolDef
from ogx_api.inference.models import (
    OpenAIAssistantMessageParam,
    OpenAIChatCompletion,
    OpenAIChatCompletionResponseMessage,
    OpenAIChatCompletionToolCall,
    OpenAIChatCompletionToolCallFunction,
    OpenAIChatCompletionUsage,
    OpenAIChoice,
)
from ogx_api.openai_responses import (
    OpenAIResponseInputToolMCP,
    OpenAIResponseReasoning,
)


@pytest.fixture
def mock_safety_api():
    safety_api = AsyncMock()
    # Mock the routing table and shields list for guardrails lookup
    safety_api.routing_table = AsyncMock()
    shield = AsyncMock()
    shield.identifier = "llama-guard"
    shield.provider_resource_id = "llama-guard-model"
    safety_api.routing_table.list_shields.return_value = AsyncMock(data=[shield])
    # Mock run_moderation to return non-flagged result by default
    safety_api.run_moderation.return_value = AsyncMock(flagged=False)
    return safety_api


@pytest.fixture
def mock_inference_api():
    inference_api = AsyncMock()
    return inference_api


@pytest.fixture
def mock_context():
    context = AsyncMock(spec=ChatCompletionContext)
    # Add required attributes that StreamingResponseOrchestrator expects
    context.tool_context = AsyncMock()
    context.tool_context.previous_tools = {}
    context.messages = []
    return context


def test_convert_tooldef_to_chat_tool_preserves_items_field():
    """Test that array parameters preserve the items field during conversion.

    This test ensures that when converting ToolDef with array-type parameters
    to OpenAI ChatCompletionToolParam format, the 'items' field is preserved.
    Without this fix, array parameters would be missing schema information about their items.
    """
    tool_def = ToolDef(
        name="test_tool",
        description="A test tool with array parameter",
        input_schema={
            "type": "object",
            "properties": {"tags": {"type": "array", "description": "List of tags", "items": {"type": "string"}}},
            "required": ["tags"],
        },
    )

    result = convert_tooldef_to_chat_tool(tool_def)

    assert result["type"] == "function"
    assert result["function"]["name"] == "test_tool"

    tags_param = result["function"]["parameters"]["properties"]["tags"]
    assert tags_param["type"] == "array"
    assert "items" in tags_param, "items field should be preserved for array parameters"
    assert tags_param["items"] == {"type": "string"}


# ---------------------------------------------------------------------------
# _separate_tool_calls regression tests
# See: https://github.com/ogx-ai/ogx/issues/5301
# ---------------------------------------------------------------------------


def _make_mcp_server(**kwargs) -> OpenAIResponseInputToolMCP:
    defaults = {"server_label": "test-server", "server_url": "http://localhost:9999/mcp"}
    defaults.update(kwargs)
    return OpenAIResponseInputToolMCP(**defaults)


def _make_tool_call(call_id: str, name: str, arguments: str = "{}") -> OpenAIChatCompletionToolCall:
    return OpenAIChatCompletionToolCall(
        id=call_id,
        function=OpenAIChatCompletionToolCallFunction(name=name, arguments=arguments),
    )


def _build_orchestrator(mcp_tool_to_server: dict[str, OpenAIResponseInputToolMCP]) -> StreamingResponseOrchestrator:
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
    mock_ctx.response_tools = [
        MagicMock(type="mcp", name="get_weather"),
        MagicMock(type="mcp", name="get_time"),
        MagicMock(type="mcp", name="get_news"),
    ]
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
        safety_api=None,
    )


def _make_response(tool_calls: list[OpenAIChatCompletionToolCall]):
    """Build a mock chat completion response with a single choice containing the given tool calls."""
    return MagicMock(
        choices=[
            OpenAIChoice(
                index=0,
                finish_reason="tool_calls",
                message=OpenAIChatCompletionResponseMessage(
                    role="assistant",
                    content=None,
                    tool_calls=tool_calls,
                ),
            )
        ]
    )


class TestAllDeferredOrDenied:
    """When all tool calls are deferred/denied, the assistant message should be fully popped."""

    def test_single_approval_pops_assistant_message(self):
        mcp_server = _make_mcp_server(require_approval="always")
        tool_map = {"get_weather": mcp_server}
        orch = _build_orchestrator(tool_map)

        tool_calls = [_make_tool_call("call_1", "get_weather")]
        response = _make_response(tool_calls)
        messages = ["system_msg", "user_msg"]

        _, _, approvals, result_messages = orch._separate_tool_calls(response, messages)

        assert len(approvals) == 1
        assert len(result_messages) == 2
        assert result_messages == ["system_msg", "user_msg"]

    def test_multiple_approvals_pops_once_not_per_tool_call(self):
        mcp_server = _make_mcp_server(require_approval="always")
        tool_map = {"get_weather": mcp_server, "get_time": mcp_server, "get_news": mcp_server}
        orch = _build_orchestrator(tool_map)

        tool_calls = [
            _make_tool_call("call_1", "get_weather"),
            _make_tool_call("call_2", "get_time"),
            _make_tool_call("call_3", "get_news"),
        ]
        response = _make_response(tool_calls)
        messages = ["system_msg", "user_msg"]

        _, _, approvals, result_messages = orch._separate_tool_calls(response, messages)

        assert len(approvals) == 3
        assert len(result_messages) == 2, (
            f"Expected 2 messages (original preserved), got {len(result_messages)}. "
            "The pop() bug is removing more than the assistant message."
        )
        assert result_messages == ["system_msg", "user_msg"]

    def test_two_approvals_does_not_eat_user_message(self):
        mcp_server = _make_mcp_server(require_approval="always")
        tool_map = {"get_weather": mcp_server, "get_time": mcp_server}
        orch = _build_orchestrator(tool_map)

        tool_calls = [
            _make_tool_call("call_1", "get_weather"),
            _make_tool_call("call_2", "get_time"),
        ]
        response = _make_response(tool_calls)
        messages = ["system_msg", "user_msg"]

        _, _, approvals, result_messages = orch._separate_tool_calls(response, messages)

        assert len(approvals) == 2
        assert "user_msg" in result_messages
        assert "system_msg" in result_messages

    def test_all_denied_pops_assistant_message(self):
        mcp_server = _make_mcp_server(require_approval="always")
        tool_map = {"get_weather": mcp_server, "get_time": mcp_server}
        orch = _build_orchestrator(tool_map)

        denial = MagicMock()
        denial.approve = False
        orch.ctx.approval_response = MagicMock(return_value=denial)

        tool_calls = [
            _make_tool_call("call_1", "get_weather"),
            _make_tool_call("call_2", "get_time"),
        ]
        response = _make_response(tool_calls)
        messages = ["system_msg", "user_msg"]

        _, _, approvals, result_messages = orch._separate_tool_calls(response, messages)

        assert len(approvals) == 0
        assert len(result_messages) == 2
        assert result_messages == ["system_msg", "user_msg"]


class TestMixedApproval:
    """When some tool calls are executed and some deferred/denied, the assistant
    message should be replaced with one containing only the executed tool calls."""

    def test_mix_replaces_assistant_message_with_executed_only(self):
        mcp_server = _make_mcp_server(require_approval="always")
        tool_map = {"get_weather": mcp_server, "get_time": mcp_server}
        orch = _build_orchestrator(tool_map)

        approval = MagicMock()
        approval.approve = True

        def side_effect(name, args):
            if name == "get_weather":
                return approval
            return None

        orch.ctx.approval_response = MagicMock(side_effect=side_effect)

        tc_weather = _make_tool_call("call_1", "get_weather")
        tc_time = _make_tool_call("call_2", "get_time")
        response = _make_response([tc_weather, tc_time])
        messages = ["system_msg", "user_msg"]

        _, non_function, approvals, result_messages = orch._separate_tool_calls(response, messages)

        assert len(non_function) == 1
        assert non_function[0].id == "call_1"
        assert len(approvals) == 1
        assert approvals[0].id == "call_2"

        assert len(result_messages) == 3
        assert result_messages[0] == "system_msg"
        assert result_messages[1] == "user_msg"

        replaced_msg = result_messages[2]
        assert isinstance(replaced_msg, OpenAIAssistantMessageParam)
        assert len(replaced_msg.tool_calls) == 1
        assert replaced_msg.tool_calls[0].id == "call_1"

    def test_mix_with_two_executed_one_deferred(self):
        always_server = _make_mcp_server(require_approval="always")
        never_server = _make_mcp_server(require_approval="never")
        tool_map = {"get_weather": never_server, "get_time": never_server, "get_news": always_server}
        orch = _build_orchestrator(tool_map)

        tc_weather = _make_tool_call("call_1", "get_weather")
        tc_time = _make_tool_call("call_2", "get_time")
        tc_news = _make_tool_call("call_3", "get_news")
        response = _make_response([tc_weather, tc_time, tc_news])
        messages = ["system_msg", "user_msg"]

        _, non_function, approvals, result_messages = orch._separate_tool_calls(response, messages)

        assert len(non_function) == 2
        assert len(approvals) == 1
        assert approvals[0].id == "call_3"

        replaced_msg = result_messages[2]
        assert isinstance(replaced_msg, OpenAIAssistantMessageParam)
        assert len(replaced_msg.tool_calls) == 2
        tool_call_ids = {tc.id for tc in replaced_msg.tool_calls}
        assert tool_call_ids == {"call_1", "call_2"}

    def test_mix_denied_and_executed_replaces_correctly(self):
        mcp_server = _make_mcp_server(require_approval="always")
        tool_map = {"get_weather": mcp_server, "get_time": mcp_server}
        orch = _build_orchestrator(tool_map)

        approval = MagicMock()
        approval.approve = True
        denial = MagicMock()
        denial.approve = False

        def side_effect(name, args):
            if name == "get_weather":
                return approval
            return denial

        orch.ctx.approval_response = MagicMock(side_effect=side_effect)

        tc_weather = _make_tool_call("call_1", "get_weather")
        tc_time = _make_tool_call("call_2", "get_time")
        response = _make_response([tc_weather, tc_time])
        messages = ["system_msg", "user_msg"]

        _, non_function, approvals, result_messages = orch._separate_tool_calls(response, messages)

        assert len(non_function) == 1
        assert len(approvals) == 0

        replaced_msg = result_messages[2]
        assert isinstance(replaced_msg, OpenAIAssistantMessageParam)
        assert len(replaced_msg.tool_calls) == 1
        assert replaced_msg.tool_calls[0].id == "call_1"

    def test_original_messages_always_preserved(self):
        mcp_server = _make_mcp_server(require_approval="always")
        tool_map = {"get_weather": mcp_server, "get_time": mcp_server, "get_news": mcp_server}
        orch = _build_orchestrator(tool_map)

        approval = MagicMock()
        approval.approve = True

        def side_effect(name, args):
            if name == "get_weather":
                return approval
            return None

        orch.ctx.approval_response = MagicMock(side_effect=side_effect)

        tool_calls = [
            _make_tool_call("call_1", "get_weather"),
            _make_tool_call("call_2", "get_time"),
            _make_tool_call("call_3", "get_news"),
        ]
        response = _make_response(tool_calls)
        messages = ["system_msg", "user_msg"]

        _, _, _, result_messages = orch._separate_tool_calls(response, messages)

        assert result_messages[0] == "system_msg"
        assert result_messages[1] == "user_msg"


class TestAllExecuted:
    """When all tool calls are executed, the assistant message should remain untouched."""

    def test_no_approvals_needed_keeps_full_assistant_message(self):
        mcp_server = _make_mcp_server(require_approval="never")
        tool_map = {"get_weather": mcp_server, "get_time": mcp_server}
        orch = _build_orchestrator(tool_map)

        tool_calls = [
            _make_tool_call("call_1", "get_weather"),
            _make_tool_call("call_2", "get_time"),
        ]
        response = _make_response(tool_calls)
        messages = ["system_msg", "user_msg"]

        _, non_function, approvals, result_messages = orch._separate_tool_calls(response, messages)

        assert len(non_function) == 2
        assert len(approvals) == 0
        assert len(result_messages) == 3

        assistant_msg = result_messages[2]
        assert isinstance(assistant_msg, OpenAIAssistantMessageParam)
        assert len(assistant_msg.tool_calls) == 2

    def test_all_pre_approved_keeps_full_assistant_message(self):
        mcp_server = _make_mcp_server(require_approval="always")
        tool_map = {"get_weather": mcp_server, "get_time": mcp_server}
        orch = _build_orchestrator(tool_map)

        approval = MagicMock()
        approval.approve = True
        orch.ctx.approval_response = MagicMock(return_value=approval)

        tool_calls = [
            _make_tool_call("call_1", "get_weather"),
            _make_tool_call("call_2", "get_time"),
        ]
        response = _make_response(tool_calls)
        messages = ["system_msg", "user_msg"]

        _, non_function, approvals, result_messages = orch._separate_tool_calls(response, messages)

        assert len(non_function) == 2
        assert len(approvals) == 0
        assert len(result_messages) == 3

        assistant_msg = result_messages[2]
        assert isinstance(assistant_msg, OpenAIAssistantMessageParam)
        assert len(assistant_msg.tool_calls) == 2


# ---------------------------------------------------------------------------
# Reasoning summary tests
# ---------------------------------------------------------------------------


class TestShouldSummarizeReasoning:
    def test_returns_false_when_reasoning_is_none(self):
        assert should_summarize_reasoning(None) is False

    def test_returns_true_for_concise(self):
        reasoning = OpenAIResponseReasoning(summary="concise")
        assert should_summarize_reasoning(reasoning) is True

    def test_returns_true_for_detailed(self):
        reasoning = OpenAIResponseReasoning(summary="detailed")
        assert should_summarize_reasoning(reasoning) is True

    def test_returns_true_for_auto(self):
        reasoning = OpenAIResponseReasoning(summary="auto")
        assert should_summarize_reasoning(reasoning) is True


class TestBuildSummaryPrompt:
    def test_concise_prompt_asks_for_short_summary(self):
        prompt = build_summary_prompt("Some reasoning text", "concise")
        assert "one or two sentences" in prompt
        assert "Some reasoning text" in prompt

    def test_detailed_prompt_preserves_logical_steps(self):
        prompt = build_summary_prompt("Some reasoning text", "detailed")
        assert "Preserve the key logical steps" in prompt
        assert "Some reasoning text" in prompt

    def test_auto_falls_through_to_concise(self):
        prompt_auto = build_summary_prompt("text", "auto")
        prompt_concise = build_summary_prompt("text", "concise")
        assert prompt_auto == prompt_concise


def _make_completion(content: str, usage: OpenAIChatCompletionUsage | None = None) -> OpenAIChatCompletion:
    """Build a mock non-streaming chat completion response."""
    return OpenAIChatCompletion(
        id="comp_1",
        choices=[
            OpenAIChoice(
                index=0,
                finish_reason="stop",
                message=OpenAIChatCompletionResponseMessage(content=content),
            )
        ],
        created=0,
        model="test-model",
        object="chat.completion",
        usage=usage,
    )


class TestSummarizeReasoning:
    async def test_returns_summary_text(self):
        mock_inference = AsyncMock()
        mock_inference.openai_chat_completion.return_value = _make_completion("The answer is 4.")

        result = await summarize_reasoning(
            inference_api=mock_inference,
            model="test-model",
            reasoning_text="Simple math.",
            summary_mode="concise",
        )

        assert result == "The answer is 4."

    async def test_returns_none_for_empty_content(self):
        mock_inference = AsyncMock()
        mock_inference.openai_chat_completion.return_value = _make_completion("")

        result = await summarize_reasoning(
            inference_api=mock_inference,
            model="test-model",
            reasoning_text="reasoning",
            summary_mode="concise",
        )

        assert result is None

    async def test_collects_usage(self):
        usage_data = OpenAIChatCompletionUsage(
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
        )
        mock_inference = AsyncMock()
        mock_inference.openai_chat_completion.return_value = _make_completion("summary", usage=usage_data)

        summary_usage: list[OpenAIChatCompletionUsage] = []
        result = await summarize_reasoning(
            inference_api=mock_inference,
            model="test-model",
            reasoning_text="reasoning",
            summary_mode="concise",
            summary_usage=summary_usage,
        )

        assert result == "summary"
        assert len(summary_usage) == 1
        assert summary_usage[0].prompt_tokens == 10
        assert summary_usage[0].completion_tokens == 5

    async def test_preserves_multiline_content(self):
        full_content = "First paragraph.\n\nSecond paragraph."
        mock_inference = AsyncMock()
        mock_inference.openai_chat_completion.return_value = _make_completion(full_content)

        result = await summarize_reasoning(
            inference_api=mock_inference,
            model="test-model",
            reasoning_text="complex reasoning",
            summary_mode="detailed",
        )

        assert result == full_content

    async def test_inference_failure_raises(self):
        mock_inference = AsyncMock()
        mock_inference.openai_chat_completion.side_effect = RuntimeError("provider down")

        with pytest.raises(RuntimeError, match="provider down"):
            await summarize_reasoning(
                inference_api=mock_inference,
                model="test-model",
                reasoning_text="reasoning",
                summary_mode="concise",
            )

    async def test_unexpected_streaming_response_raises(self):
        mock_inference = AsyncMock()
        mock_inference.openai_chat_completion.return_value = MagicMock(spec=AsyncIterator)

        with pytest.raises(RuntimeError, match="Expected non-streaming response"):
            await summarize_reasoning(
                inference_api=mock_inference,
                model="test-model",
                reasoning_text="reasoning",
                summary_mode="concise",
            )

    async def test_uses_correct_summary_mode(self):
        mock_inference = AsyncMock()
        mock_inference.openai_chat_completion.return_value = _make_completion("summary")

        await summarize_reasoning(
            inference_api=mock_inference,
            model="test-model",
            reasoning_text="some reasoning",
            summary_mode="detailed",
        )

        call_args = mock_inference.openai_chat_completion.call_args[0][0]
        user_msg = call_args.messages[1].content
        assert "Preserve the key logical steps" in user_msg
