# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Regression tests for the second round of code review on #6838's compaction fixes
(branch fix-compaction-order-and-hidden-response, commit 677d206 and after):

- A deleted conversation's stored message cache outlives the conversation itself, so
  skipping the raw item log once that cache exists let a turn on a deleted
  conversation proceed instead of 404ing.
- A conversation= tool-result turn crossing the compaction threshold sent the
  summarizer a message history ending on a dangling tool_calls message with no
  reply, which some backends reject.
- summary_max_output_tokens defaulted to 1024, too small for reasoning models, and an
  empty-because-truncated summary was indistinguishable from any other empty summary
  and silently discarded its usage.
- The manual /compact endpoint ignored summary_max_output_tokens and surfaced a raw
  OpenAI-param-named validation error for a bad max_output_tokens.
- conversation= mode's compacted output dropped every retained verbatim user message
  that previous_response_id mode keeps.
- A guardrail refusal response never carried any usage at all, compaction or
  otherwise, since _create_refusal_response built its OpenAIResponseObject without
  a usage field.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from ogx.providers.inline.responses.builtin.responses.streaming import StreamingResponseOrchestrator
from ogx.providers.inline.responses.builtin.responses.types import ChatCompletionContext
from ogx_api import ConversationNotFoundError, InvalidParameterError
from ogx_api.inference import OpenAIChatCompletionUsage
from ogx_api.openai_responses import (
    OpenAIResponseCompaction,
    OpenAIResponseInputFunctionToolCallOutput,
)
from ogx_api.responses.models import CreateResponseRequest
from tests.unit.providers.responses.builtin.compaction_test_helpers import (
    _compaction_completion,
    _dual_inference_mock,
    _previous_response,
)


class TestDeletedConversationStillFailsFast:
    """A deleted conversation's responses_store.conversation_messages row is not
    cleaned up by conversation deletion itself (a layering boundary: the core
    Conversations provider has no dependency on this Responses-specific cache
    table), so skipping the raw item log once that cache exists must not also skip
    checking the conversation still exists.
    """

    async def test_turn_on_a_deleted_conversation_raises_instead_of_succeeding(
        self, openai_responses_impl, mock_inference_api, mock_responses_store, mock_conversations_api
    ):
        from ogx_api import OpenAIUserMessageParam

        mock_responses_store.get_conversation_messages.return_value = [OpenAIUserMessageParam(content="Hi")]
        mock_conversations_api.get_conversation.side_effect = ConversationNotFoundError("conv_deleted")
        mock_inference_api.openai_chat_completion = AsyncMock()

        with pytest.raises(ConversationNotFoundError):
            await openai_responses_impl.create_openai_response(
                CreateResponseRequest(
                    input="Are you still there?",
                    model="meta-llama/Llama-3.1-8B-Instruct",
                    conversation="conv_272fa1af1b67b662ee7ef05f72ec300d92425257645a531a",
                )
            )

        mock_inference_api.openai_chat_completion.assert_not_called()

    async def test_turn_on_a_deleted_conversation_clears_the_stale_message_cache(
        self, openai_responses_impl, mock_inference_api, mock_responses_store, mock_conversations_api
    ):
        from ogx_api import OpenAIUserMessageParam

        mock_responses_store.get_conversation_messages.return_value = [OpenAIUserMessageParam(content="Hi")]
        mock_conversations_api.get_conversation.side_effect = ConversationNotFoundError("conv_deleted")
        mock_inference_api.openai_chat_completion = AsyncMock()

        with pytest.raises(ConversationNotFoundError):
            await openai_responses_impl.create_openai_response(
                CreateResponseRequest(
                    input="Are you still there?",
                    model="meta-llama/Llama-3.1-8B-Instruct",
                    conversation="conv_272fa1af1b67b662ee7ef05f72ec300d92425257645a531a",
                )
            )

        mock_responses_store.delete_conversation_messages.assert_awaited_once_with(
            "conv_272fa1af1b67b662ee7ef05f72ec300d92425257645a531a"
        )

    async def test_existing_conversation_is_unaffected(
        self, openai_responses_impl, mock_inference_api, mock_responses_store, mock_conversations_api
    ):
        """The existence check must stay cheap and silent for the common case: a
        conversation that still exists, with a stored message log already in place.
        """
        from ogx_api import OpenAIUserMessageParam

        mock_responses_store.get_conversation_messages.return_value = [OpenAIUserMessageParam(content="Hi")]
        mock_inference_api.openai_chat_completion = _dual_inference_mock()

        await openai_responses_impl.create_openai_response(
            CreateResponseRequest(
                input="Still here?",
                model="meta-llama/Llama-3.1-8B-Instruct",
                conversation="conv_272fa1af1b67b662ee7ef05f72ec300d92425257645a531a",
            )
        )

        mock_conversations_api.get_conversation.assert_awaited_once()
        mock_responses_store.delete_conversation_messages.assert_not_called()


class TestConversationModeToolResultSurvivesCompactionInTheSummarizerCallToo:
    """The open-call carryover into the final compacted input (already fixed) is not
    enough on its own for conversation= mode: the summarization call itself must also
    not see a dangling tool_calls message with no reply, or backends that enforce
    tool-call pairing reject that call outright.
    """

    async def test_summarizer_messages_do_not_end_on_an_unanswered_tool_call(
        self, openai_responses_impl, mock_inference_api, mock_responses_store, mock_conversations_api
    ):
        from ogx_api import OpenAIAssistantMessageParam
        from ogx_api.inference import OpenAIChatCompletionToolCall, OpenAIChatCompletionToolCallFunction

        mock_responses_store.get_conversation_messages.return_value = [
            OpenAIAssistantMessageParam(
                tool_calls=[
                    OpenAIChatCompletionToolCall(
                        id="call_1",
                        function=OpenAIChatCompletionToolCallFunction(name="get_weather", arguments="{}"),
                    )
                ]
            ),
        ]
        mock_inference_api.openai_chat_completion = _dual_inference_mock()

        await openai_responses_impl.create_openai_response(
            CreateResponseRequest(
                input=[OpenAIResponseInputFunctionToolCallOutput(call_id="call_1", output="72F and sunny")],
                model="meta-llama/Llama-3.1-8B-Instruct",
                conversation="conv_272fa1af1b67b662ee7ef05f72ec300d92425257645a531a",
                context_management=[{"type": "compaction", "compact_threshold": 1}],
            )
        )

        compaction_call = next(
            call for call in mock_inference_api.openai_chat_completion.call_args_list if not call.args[0].stream
        )
        summarizer_messages = compaction_call.args[0].messages
        assert not any(getattr(m, "tool_calls", None) for m in summarizer_messages)


class TestSummaryMaxOutputTokensDefault:
    async def test_default_is_none_not_a_small_cap(self, openai_responses_impl):
        assert openai_responses_impl.compaction_config.summary_max_output_tokens is None

    async def test_uncapped_summary_call_reaches_a_reasoning_model_without_truncation(
        self, openai_responses_impl, mock_inference_api, mock_responses_store
    ):
        mock_responses_store.get_response_object.return_value = _previous_response()
        mock_inference_api.openai_chat_completion = _dual_inference_mock()

        await openai_responses_impl.create_openai_response(
            CreateResponseRequest(
                input="What is the code word?",
                model="meta-llama/Llama-3.1-8B-Instruct",
                previous_response_id="resp-previous-123",
                context_management=[{"type": "compaction", "compact_threshold": 1}],
            )
        )

        compaction_call = next(
            call for call in mock_inference_api.openai_chat_completion.call_args_list if not call.args[0].stream
        )
        assert compaction_call.args[0].max_completion_tokens is None


class TestTruncatedToEmptySummaryKeepsItsUsage:
    """A reasoning model that spends its whole output-token cap thinking, and never
    writes a summary, must still be distinguishable (via the finish_reason=="length"
    warning) from any other empty-summary cause, and its wasted tokens must still be
    accounted for rather than silently dropped.
    """

    async def test_empty_summary_truncated_by_length_still_returns_its_usage(
        self, openai_responses_impl, mock_inference_api, mock_responses_store, caplog
    ):
        mock_responses_store.get_response_object.return_value = _previous_response()

        async def side_effect(params):
            if params.stream:
                from tests.unit.providers.responses.builtin.test_openai_responses_helpers import fake_stream

                return fake_stream()
            return _compaction_completion(None, finish_reason="length", prompt_tokens=200, completion_tokens=150)

        mock_inference_api.openai_chat_completion = AsyncMock(side_effect=side_effect)

        with caplog.at_level("WARNING"):
            result = await openai_responses_impl.create_openai_response(
                CreateResponseRequest(
                    input="What is the code word?",
                    model="meta-llama/Llama-3.1-8B-Instruct",
                    previous_response_id="resp-previous-123",
                    context_management=[{"type": "compaction", "compact_threshold": 1}],
                )
            )

        assert "own output token cap" in caplog.text
        # The wasted summarization call's tokens must still show up, not be dropped
        # just because the summary itself came back empty.
        assert result.usage.input_tokens >= 200
        assert result.usage.output_tokens >= 150


class TestAutoCompactionSummarizerExceptionFallsBackInsteadOfFailingTheTurn:
    async def test_summarizer_call_raising_keeps_the_turn_working(
        self, openai_responses_impl, mock_inference_api, mock_responses_store
    ):
        mock_responses_store.get_response_object.return_value = _previous_response()

        async def side_effect(params):
            if params.stream:
                from tests.unit.providers.responses.builtin.test_openai_responses_helpers import fake_stream

                return fake_stream()
            raise RuntimeError("summarization model is over capacity")

        mock_inference_api.openai_chat_completion = AsyncMock(side_effect=side_effect)

        result = await openai_responses_impl.create_openai_response(
            CreateResponseRequest(
                input="What is the code word?",
                model="meta-llama/Llama-3.1-8B-Instruct",
                previous_response_id="resp-previous-123",
                context_management=[{"type": "compaction", "compact_threshold": 1}],
            )
        )

        assert result.status == "completed"
        assert not any(isinstance(item, OpenAIResponseCompaction) for item in result.output)


class TestManualCompactEndpointMaxOutputTokens:
    async def test_defaults_to_the_configured_summary_cap_when_unset(self, openai_responses_impl, mock_inference_api):
        openai_responses_impl.compaction_config.summary_max_output_tokens = 777
        mock_inference_api.openai_chat_completion = _dual_inference_mock()

        await openai_responses_impl.compact_openai_response(model="test-model", input="Some content.")

        params = mock_inference_api.openai_chat_completion.call_args.args[0]
        assert params.max_completion_tokens == 777

    @pytest.mark.parametrize("bad_value", ["abc", 0, -5])
    async def test_bad_value_raises_invalid_parameter_error_named_after_itself(
        self, openai_responses_impl, mock_inference_api, bad_value
    ):
        mock_inference_api.openai_chat_completion = _dual_inference_mock()

        with pytest.raises(InvalidParameterError) as exc_info:
            await openai_responses_impl.compact_openai_response(
                model="test-model", input="Some content.", max_output_tokens=bad_value
            )

        assert "max_output_tokens" in str(exc_info.value)
        assert "max_completion_tokens" not in str(exc_info.value)
        mock_inference_api.openai_chat_completion.assert_not_called()


class TestConversationModeKeepsVerbatimUserMessages:
    """previous_response_id mode keeps every retained user message verbatim in the
    compacted output (OpenAI's documented compaction behavior); conversation= mode
    must line up with it, since both now compact through the same _build_compaction.
    """

    async def test_plain_string_user_messages_survive_conversation_mode_compaction(
        self, openai_responses_impl, mock_inference_api, mock_responses_store, mock_conversations_api
    ):
        from ogx_api import OpenAIAssistantMessageParam, OpenAIUserMessageParam

        mock_responses_store.get_conversation_messages.return_value = [
            OpenAIUserMessageParam(content="My name is Alex."),
            OpenAIAssistantMessageParam(content="Nice to meet you, Alex."),
        ]
        mock_inference_api.openai_chat_completion = _dual_inference_mock()

        result = await openai_responses_impl.create_openai_response(
            CreateResponseRequest(
                input="What is my name?",
                model="meta-llama/Llama-3.1-8B-Instruct",
                conversation="conv_272fa1af1b67b662ee7ef05f72ec300d92425257645a531a",
                context_management=[{"type": "compaction", "compact_threshold": 1}],
            )
        )

        assert result.status == "completed"
        main_call = next(
            call for call in mock_inference_api.openai_chat_completion.call_args_list if call.args[0].stream
        )
        messages = main_call.args[0].messages

        def _message_text(message: object) -> str:
            content = getattr(message, "content", None)
            if isinstance(content, str):
                return content
            return "".join(getattr(part, "text", "") for part in content or [])

        assert any(_message_text(m) == "My name is Alex." for m in messages)


class TestGuardrailRefusalCarriesUsage:
    async def test_refusal_response_includes_accumulated_usage(self):
        context = AsyncMock(spec=ChatCompletionContext)
        context.tool_context = AsyncMock()
        context.tool_context.previous_tools = {}
        context.messages = []
        context.tool_choice = None
        context.available_tools = MagicMock(return_value=[])
        context.model = "test-model"
        context.temperature = None
        context.top_p = None
        context.frequency_penalty = None

        orchestrator = StreamingResponseOrchestrator(
            inference_api=AsyncMock(),
            ctx=context,
            response_id="resp_refusal_usage",
            created_at=0,
            text=MagicMock(),
            max_infer_iters=1,
            tool_executor=MagicMock(),
            instructions=None,
            moderation_endpoint="http://localhost:8080/v1/moderations",
            enable_guardrails=True,
            initial_usage=OpenAIChatCompletionUsage(prompt_tokens=200, completion_tokens=150, total_tokens=350),
        )

        stream_chunk = await orchestrator._create_refusal_response("blocked content")

        assert stream_chunk.response.usage is not None
        assert stream_chunk.response.usage.input_tokens == 200
        assert stream_chunk.response.usage.output_tokens == 150
