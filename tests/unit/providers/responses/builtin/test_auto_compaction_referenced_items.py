# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Automatic compaction must carry forward any history item the new turn refers back
to by id -- a function_call answered by a function_call_output, or an
mcp_approval_request answered by an mcp_approval_response -- instead of summarizing it
away and breaking the turn.
"""

from unittest.mock import AsyncMock

from ogx.providers.inline.responses.builtin.responses.openai_responses import OpenAIResponsesImpl
from ogx_api import (
    OpenAIAssistantMessageParam,
)
from ogx_api.inference import OpenAIChatCompletionToolCall, OpenAIChatCompletionToolCallFunction
from ogx_api.openai_responses import (
    OpenAIResponseCompaction,
    OpenAIResponseInputFunctionToolCallOutput,
    OpenAIResponseMCPApprovalRequest,
    OpenAIResponseMCPApprovalResponse,
    OpenAIResponseMessage,
    OpenAIResponseOutputMessageFunctionToolCall,
)
from ogx_api.responses.models import CreateResponseRequest
from tests.unit.providers.responses.builtin.compaction_test_helpers import (
    _dual_inference_mock,
    _previous_response,
)


class TestOpenToolCallSurvivesCompaction:
    """Code review of #6838: if the new input answers a function_call still sitting in
    history (a tool loop's pending call), _build_compaction's "keep only user messages
    verbatim" rule dropped that function_call -- leaving the incoming
    function_call_output with no corresponding function_call anywhere in the final
    input, which convert_response_input_to_chat_messages rejects with a 400-worthy
    ValueError. The open call must survive compaction, placed right after the
    compaction item and before the input answering it.
    """

    async def test_pending_function_call_survives_compaction_with_previous_response_id(
        self, openai_responses_impl, mock_inference_api, mock_responses_store
    ):
        previous_response = _previous_response()
        previous_response.output = [
            *previous_response.output,
            OpenAIResponseOutputMessageFunctionToolCall(
                call_id="call_abc123", name="get_weather", arguments='{"city": "Paris"}'
            ),
        ]
        # _process_input_with_previous_response always builds a "default" messages
        # list from previous_response.messages before compaction is even considered
        # (it's discarded and rebuilt from the compacted input if compaction fires,
        # but must not raise before getting there) -- it needs the matching tool-call
        # message too, mirroring the function_call item added to .output above.
        previous_response.messages = [
            *previous_response.messages,
            OpenAIAssistantMessageParam(
                tool_calls=[
                    OpenAIChatCompletionToolCall(
                        id="call_abc123",
                        function=OpenAIChatCompletionToolCallFunction(
                            name="get_weather", arguments='{"city": "Paris"}'
                        ),
                    )
                ]
            ),
        ]
        mock_responses_store.get_response_object.return_value = previous_response
        mock_inference_api.openai_chat_completion = _dual_inference_mock()

        result = await openai_responses_impl.create_openai_response(
            CreateResponseRequest(
                input=[OpenAIResponseInputFunctionToolCallOutput(call_id="call_abc123", output="72F and sunny")],
                model="meta-llama/Llama-3.1-8B-Instruct",
                previous_response_id="resp-previous-123",
                context_management=[{"type": "compaction", "compact_threshold": 1}],
            )
        )

        assert result.status == "completed"
        main_call = next(
            call for call in mock_inference_api.openai_chat_completion.call_args_list if call.args[0].stream
        )
        messages = main_call.args[0].messages
        tool_call_message = next(m for m in messages if getattr(m, "tool_calls", None))
        assert tool_call_message.tool_calls[0].id == "call_abc123"
        tool_result_message = next(m for m in messages if getattr(m, "tool_call_id", None) == "call_abc123")
        assert tool_result_message.content == "72F and sunny"

    async def test_pending_function_call_survives_compaction_in_a_single_oversized_call(
        self, openai_responses_impl, mock_inference_api
    ):
        """Same bug, but via the other path that can separate a function_call from its
        output: a single oversized first call with no previous_response_id/conversation,
        where the call-site split (everything but the last item becomes compactable
        history) happens to cut right between the call and its result.
        """
        mock_inference_api.openai_chat_completion = _dual_inference_mock()

        large_input: list = [OpenAIResponseMessage(role="user", content=f"Tell me about topic {i}.") for i in range(5)]
        large_input.append(
            OpenAIResponseOutputMessageFunctionToolCall(
                call_id="call_xyz789", name="get_weather", arguments='{"city": "Paris"}'
            )
        )
        large_input.append(OpenAIResponseInputFunctionToolCallOutput(call_id="call_xyz789", output="72F and sunny"))

        result = await openai_responses_impl.create_openai_response(
            CreateResponseRequest(
                input=large_input,
                model="meta-llama/Llama-3.1-8B-Instruct",
                context_management=[{"type": "compaction", "compact_threshold": 1}],
            )
        )

        assert result.status == "completed"
        main_call = next(
            call for call in mock_inference_api.openai_chat_completion.call_args_list if call.args[0].stream
        )
        tool_result_message = next(
            m for m in main_call.args[0].messages if getattr(m, "tool_call_id", None) == "call_xyz789"
        )
        assert tool_result_message.content == "72F and sunny"


class TestMcpApprovalRequestSurvivesCompaction:
    """An mcp_approval_response in the new turn is matched to its mcp_approval_request
    by approval_request_id, and the request itself is looked up in the turn's input by
    ChatCompletionContext. If compaction summarizes the request away, the approved tool
    call can no longer be paired with its approval and is sent back for approval again,
    so the request must be carried forward like an open function_call.
    """

    @staticmethod
    def _approval_request() -> OpenAIResponseMCPApprovalRequest:
        return OpenAIResponseMCPApprovalRequest(
            id="mcpr_abc123",
            name="get_weather",
            arguments='{"city": "Paris"}',
            server_label="weather",
        )

    async def test_approval_request_is_carried_forward_between_summary_and_new_input(self, mock_inference_api):
        from ogx.providers.inline.responses.builtin.config import CompactionConfig

        impl = object.__new__(OpenAIResponsesImpl)
        impl.inference_api = mock_inference_api
        impl.files_api = AsyncMock()
        impl.compaction_config = CompactionConfig()
        mock_inference_api.openai_chat_completion = _dual_inference_mock()

        approval_request = self._approval_request()
        history = [
            OpenAIResponseMessage(role="user", content="What's the weather in Paris?"),
            approval_request,
        ]
        new_input = [OpenAIResponseMCPApprovalResponse(approval_request_id="mcpr_abc123", approve=True)]

        effective_input, compaction_item, _ = await impl._maybe_auto_compact(
            history, new_input, "test-model", [{"type": "compaction", "compact_threshold": 1}]
        )

        assert compaction_item is not None
        assert effective_input[-1] is new_input[0]
        assert effective_input[-2] is approval_request
        assert effective_input.index(compaction_item) < effective_input.index(approval_request)

    async def test_approval_still_resolves_for_the_approved_call_after_compaction(self, mock_inference_api):
        from ogx.providers.inline.responses.builtin.config import CompactionConfig
        from ogx.providers.inline.responses.builtin.responses.types import ChatCompletionContext
        from ogx_api.inference.models import OpenAIResponseFormatText

        impl = object.__new__(OpenAIResponsesImpl)
        impl.inference_api = mock_inference_api
        impl.files_api = AsyncMock()
        impl.compaction_config = CompactionConfig()
        mock_inference_api.openai_chat_completion = _dual_inference_mock()

        history = [
            OpenAIResponseMessage(role="user", content="What's the weather in Paris?"),
            self._approval_request(),
        ]
        new_input = [OpenAIResponseMCPApprovalResponse(approval_request_id="mcpr_abc123", approve=True)]

        effective_input, _, _ = await impl._maybe_auto_compact(
            history, new_input, "test-model", [{"type": "compaction", "compact_threshold": 1}]
        )

        ctx = ChatCompletionContext(
            model="test-model",
            messages=[],
            response_tools=None,
            temperature=None,
            top_p=None,
            response_format=OpenAIResponseFormatText(),
            tool_context=None,
            inputs=effective_input,
        )
        approval = ctx.approval_response("get_weather", '{"city": "Paris"}')
        assert approval is not None
        assert approval.approve is True

    async def test_approval_request_survives_compaction_with_previous_response_id(
        self, openai_responses_impl, mock_inference_api, mock_responses_store
    ):
        previous_response = _previous_response()
        previous_response.output = [*previous_response.output, self._approval_request()]
        mock_responses_store.get_response_object.return_value = previous_response
        mock_inference_api.openai_chat_completion = _dual_inference_mock()

        await openai_responses_impl.create_openai_response(
            CreateResponseRequest(
                input=[OpenAIResponseMCPApprovalResponse(approval_request_id="mcpr_abc123", approve=True)],
                model="meta-llama/Llama-3.1-8B-Instruct",
                previous_response_id="resp-previous-123",
                context_management=[{"type": "compaction", "compact_threshold": 1}],
            )
        )

        stored_input = mock_responses_store.upsert_response_object.call_args.kwargs["input"]
        assert any(isinstance(item, OpenAIResponseCompaction) for item in stored_input)
        assert any(
            isinstance(item, OpenAIResponseMCPApprovalRequest) and item.id == "mcpr_abc123" for item in stored_input
        )
