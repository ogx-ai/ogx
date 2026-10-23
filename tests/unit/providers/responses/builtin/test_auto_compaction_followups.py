# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Regression tests for the code-review follow-up fixes on top of #6829's automatic
compaction fix (branch fix-compaction-order-and-hidden-response): the compaction
summary could eat into the answer's own max_output_tokens budget; a pending tool
call could get dropped out from under its about-to-arrive result; conversation= mode
kept re-summarizing its entire ever-growing raw history after the first compaction;
a compacting turn stored the compaction item twice (once in its own .input, once via
.output); and the manual /compact endpoint could spread arbitrary stray request
fields into the summarization call. Split from test_auto_compaction.py to stay under
the repo's per-file line limit.
"""

from unittest.mock import AsyncMock

from ogx_api import OpenAIAssistantMessageParam, OpenAIUserMessageParam
from ogx_api.inference import OpenAIChatCompletionToolCall, OpenAIChatCompletionToolCallFunction
from ogx_api.openai_responses import (
    OpenAIResponseCompaction,
    OpenAIResponseInputFunctionToolCallOutput,
    OpenAIResponseMessage,
    OpenAIResponseObjectStreamResponseOutputItemAdded,
    OpenAIResponseObjectStreamResponseOutputItemDone,
    OpenAIResponseOutputMessageFunctionToolCall,
)
from ogx_api.responses.models import CreateResponseRequest
from tests.unit.providers.responses.builtin.compaction_test_helpers import (
    _dual_inference_mock,
    _previous_response,
)
from tests.unit.providers.responses.builtin.test_openai_responses_helpers import fake_stream


class TestConversationRawItemLogFetchedOnlyWhenNeeded:
    """Code review of #6838: every conversation= request used to fetch the full,
    possibly multi-page conversation item log (_fetch_full_conversation_items), even
    when context_management wasn't configured at all -- paying for pagination that
    only auto-compaction ever needed. Once a stored message log exists (every turn
    after the conversation's first), nothing needs the raw item log any more: it's
    never the first turn, and compaction now compacts from the stored message log
    instead (TestConversationModeCompactionStaysBounded). The raw item log is only
    still fetched, paginating as needed, when no stored message log exists yet --
    either the conversation's first turn, or a degraded case with no messages cache.
    """

    def _conversation_pages(self, total_items: int, page_size: int = 100):
        from ogx_api.conversations.models import ConversationItemList

        items = [OpenAIResponseMessage(id=f"item-{i}", role="user", content=f"Message {i}") for i in range(total_items)]
        pages = []
        for start in range(0, total_items, page_size):
            chunk = items[start : start + page_size]
            has_more = start + page_size < total_items
            pages.append(
                ConversationItemList(
                    data=chunk,
                    first_id=chunk[0].id if chunk else None,
                    last_id=chunk[-1].id if chunk else None,
                    has_more=has_more,
                )
            )
        return pages

    async def test_no_fetch_at_all_once_a_stored_message_log_exists(
        self, openai_responses_impl, mock_inference_api, mock_responses_store, mock_conversations_api
    ):
        """Even without context_management configured, a turn with a stored message
        log already in place must not touch conversations_api.list_items at all.
        """

        mock_responses_store.get_conversation_messages.return_value = [OpenAIUserMessageParam(content="Hi")]
        mock_inference_api.openai_chat_completion.return_value = fake_stream()

        await openai_responses_impl.create_openai_response(
            CreateResponseRequest(
                input="A new question.",
                model="meta-llama/Llama-3.1-8B-Instruct",
                conversation="conv_272fa1af1b67b662ee7ef05f72ec300d92425257645a531a",
            )
        )

        mock_conversations_api.list_items.assert_not_called()

    async def test_falls_back_to_the_full_paginated_fetch_without_a_stored_message_log(
        self, openai_responses_impl, mock_inference_api, mock_responses_store, mock_conversations_api
    ):
        """No stored message log (e.g. a degraded case: older data predating the
        messages cache) still needs the raw item log, paginated past the default page
        size, to reconstruct the chat history correctly.
        """
        pages = self._conversation_pages(total_items=25, page_size=10)
        mock_conversations_api.list_items = AsyncMock(side_effect=pages)
        mock_responses_store.get_conversation_messages.return_value = None
        mock_inference_api.openai_chat_completion.return_value = fake_stream()

        await openai_responses_impl.create_openai_response(
            CreateResponseRequest(
                input="A new question.",
                model="meta-llama/Llama-3.1-8B-Instruct",
                conversation="conv_272fa1af1b67b662ee7ef05f72ec300d92425257645a531a",
            )
        )

        # All 3 pages (10+10+5) must have been walked, not just the first.
        assert mock_conversations_api.list_items.call_count == 3
        main_call = mock_inference_api.openai_chat_completion.call_args
        # The reconstructed history (25 raw items) plus the new question must have
        # reached the model, not just the single new turn.
        assert len(main_call.args[0].messages) > 2


class TestCompactRequestWrapperDoesNotLeakStrayFields:
    """Code review of #6838: impl.py's compact_openai_response wrapper used to spread
    request.model_extra -- every unrecognized top-level field on the request, since
    CompactResponseRequest has extra="allow" with no typed extra_body field -- wholesale
    into the summarization call. A stray top-level field colliding with a named argument
    (e.g. max_completion_tokens, stream) raised a TypeError building the chat completion
    request, and others (e.g. service_tier) silently changed validated behavior that used
    to be ignored entirely (extra_body was accepted but never read before #6829). Mirroring
    CreateResponseRequest's own convention -- extra_body is only ever read from a nested
    "extra_body" key, never the whole model_extra dict -- fixes this.
    """

    async def test_stray_top_level_field_does_not_reach_the_chat_completion_call(self):
        from unittest.mock import AsyncMock

        from ogx.providers.inline.responses.builtin.impl import BuiltinResponsesImpl
        from ogx_api.responses.models import CompactResponseRequest

        wrapper = object.__new__(BuiltinResponsesImpl)
        wrapper.openai_responses_impl = AsyncMock()

        # A client that copy-pasted a chat-completions-style field onto the compact
        # request; it has no business reaching the summarization call at all.
        request = CompactResponseRequest(model="test-model", input="hi", max_completion_tokens=999)

        await wrapper.compact_openai_response(request)

        call_kwargs = wrapper.openai_responses_impl.compact_openai_response.call_args.kwargs
        assert call_kwargs["extra_body"] is None
        assert call_kwargs["max_output_tokens"] is None

    async def test_nested_extra_body_and_top_level_max_output_tokens_both_reach_the_impl(self):
        from unittest.mock import AsyncMock

        from ogx.providers.inline.responses.builtin.impl import BuiltinResponsesImpl
        from ogx_api.responses.models import CompactResponseRequest

        wrapper = object.__new__(BuiltinResponsesImpl)
        wrapper.openai_responses_impl = AsyncMock()

        request = CompactResponseRequest(
            model="test-model",
            input="hi",
            max_output_tokens=77,
            extra_body={"cache_salt": "xyz"},
        )

        await wrapper.compact_openai_response(request)

        call_kwargs = wrapper.openai_responses_impl.compact_openai_response.call_args.kwargs
        assert call_kwargs["max_output_tokens"] == 77
        assert call_kwargs["extra_body"] == {"cache_salt": "xyz"}

    async def test_empty_summary_raises_a_server_error_not_a_bad_request(
        self, openai_responses_impl, mock_inference_api
    ):
        """An empty summary is the summarization model's failure, not a malformed
        request -- ValueError maps to 400 (core/exceptions/mapping.py), which would
        incorrectly tell the caller their request was bad.
        """
        from ogx_api import InternalServerError

        mock_inference_api.openai_chat_completion = _dual_inference_mock(summary=None)

        try:
            await openai_responses_impl.compact_openai_response(model="test-model", input="Some content.")
        except InternalServerError as e:
            assert "empty summary" in str(e)
        else:
            raise AssertionError("expected InternalServerError for an empty summary")


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


class TestConversationModeCompactionStaysBounded:
    """Code review of #6838: conversation= mode's raw conversation item log
    (_fetch_full_conversation_items) never shrinks after a compaction -- compaction
    items are deliberately excluded from it (_sync_response_to_conversation) -- so
    using it as the compaction input/threshold source on every turn re-summarizes an
    ever-growing, uncompacted transcript every single turn after the first
    compaction, until it eventually exceeds the model's own context and the
    summarization call itself fails (it has no truncation). Compacting from the
    stored chat-message log instead (which already reflects any earlier compaction)
    keeps each compaction's input bounded to what's new since the last one.
    """

    def _conversation_pages(self, total_items: int, page_size: int = 100):
        from ogx_api.conversations.models import ConversationItemList

        items = [OpenAIResponseMessage(id=f"item-{i}", role="user", content=f"Message {i}") for i in range(total_items)]
        pages = []
        for start in range(0, total_items, page_size):
            chunk = items[start : start + page_size]
            has_more = start + page_size < total_items
            pages.append(
                ConversationItemList(
                    data=chunk,
                    first_id=chunk[0].id if chunk else None,
                    last_id=chunk[-1].id if chunk else None,
                    has_more=has_more,
                )
            )
        return pages

    async def test_recompaction_uses_the_stored_message_log_not_the_ever_growing_raw_items(
        self, openai_responses_impl, mock_inference_api, mock_responses_store, mock_conversations_api
    ):
        from ogx_api import OpenAIAssistantMessageParam

        # The raw conversation item log has already grown large, as if several turns
        # happened after an earlier compaction -- since compaction items are never
        # synced into it, it keeps every raw item forever and requires pagination.
        pages = self._conversation_pages(total_items=150, page_size=100)
        mock_conversations_api.list_items = AsyncMock(side_effect=pages)
        # But the stored chat-message log already reflects that earlier compaction: a
        # short, bounded log, not the 150-item raw transcript.
        mock_responses_store.get_conversation_messages.return_value = [
            OpenAIAssistantMessageParam(content="Summary of everything before this point."),
            OpenAIUserMessageParam(content="A recent question."),
            OpenAIAssistantMessageParam(content="A recent answer."),
        ]
        mock_inference_api.openai_chat_completion = _dual_inference_mock()

        await openai_responses_impl.create_openai_response(
            CreateResponseRequest(
                input="What did we discuss?",
                model="meta-llama/Llama-3.1-8B-Instruct",
                conversation="conv_272fa1af1b67b662ee7ef05f72ec300d92425257645a531a",
                context_management=[{"type": "compaction", "compact_threshold": 1}],
            )
        )

        compaction_call = next(
            call for call in mock_inference_api.openai_chat_completion.call_args_list if not call.args[0].stream
        )
        # 3 stored messages + the summarization prompt _build_compaction appends --
        # not the 150-item raw transcript.
        assert len(compaction_call.args[0].messages) == 4


class TestCompactionItemNotDuplicatedInNextHistory:
    """Code review of #6838: a compacting turn stored the compaction item in BOTH its
    own .input (as the last element of the compacted history) and its .output
    (surfaced as output[0]) -- so the next previous_response_id turn's history
    (previous.input + previous.output, _process_input_with_previous_response) counted
    it twice: double-counting its tokens, re-summarizing it twice on the next
    compaction, and listing it twice via the input_items endpoint.
    """

    async def test_compacting_turn_does_not_store_the_compaction_item_in_its_own_input(
        self, openai_responses_impl, mock_inference_api, mock_responses_store
    ):
        mock_responses_store.get_response_object.return_value = _previous_response()
        mock_inference_api.openai_chat_completion = _dual_inference_mock()

        result = await openai_responses_impl.create_openai_response(
            CreateResponseRequest(
                input="What is the code word?",
                model="meta-llama/Llama-3.1-8B-Instruct",
                previous_response_id="resp-previous-123",
                context_management=[{"type": "compaction", "compact_threshold": 1}],
            )
        )

        assert isinstance(result.output[0], OpenAIResponseCompaction)

        final_store_call = next(
            call
            for call in mock_responses_store.upsert_response_object.call_args_list
            if call.kwargs["response_object"].status == "completed"
        )
        stored_input = final_store_call.kwargs["input"]
        assert not any(isinstance(item, OpenAIResponseCompaction) for item in stored_input)


class TestAutoCompactionEndToEndFollowups:
    """Streaming-events and multi-turn chaining checks requested in code review of
    #6838's Test Plan section, exercising the real streaming path and a second
    previous_response_id turn built on a compacting one -- not just the final
    response object a single call produces.
    """

    async def test_streaming_events_surface_the_compaction_item(
        self, openai_responses_impl, mock_inference_api, mock_responses_store
    ):
        """Point 9, streaming variant: a non-streaming client only ever sees the final
        response.output (covered above), but a streaming client needs the
        response.output_item.added/done events for the compaction item itself -- and
        needs to see them before any event for the actual answer, matching the
        ordering guarantee tested directly against _maybe_auto_compact.
        """

        mock_responses_store.get_response_object.return_value = _previous_response()
        mock_inference_api.openai_chat_completion = _dual_inference_mock()

        stream = await openai_responses_impl.create_openai_response(
            CreateResponseRequest(
                input="What is the code word?",
                model="meta-llama/Llama-3.1-8B-Instruct",
                previous_response_id="resp-previous-123",
                context_management=[{"type": "compaction", "compact_threshold": 1}],
                stream=True,
            )
        )

        events = [chunk async for chunk in stream]
        added_index = next(
            i
            for i, e in enumerate(events)
            if isinstance(e, OpenAIResponseObjectStreamResponseOutputItemAdded)
            and isinstance(e.item, OpenAIResponseCompaction)
        )
        done_index = next(
            i
            for i, e in enumerate(events)
            if isinstance(e, OpenAIResponseObjectStreamResponseOutputItemDone)
            and isinstance(e.item, OpenAIResponseCompaction)
        )
        assert added_index < done_index
        # No other output item (e.g. the actual answer) may be added before the
        # compaction item's own added/done pair.
        other_output_item_indices = [
            i
            for i, e in enumerate(events)
            if isinstance(e, OpenAIResponseObjectStreamResponseOutputItemAdded)
            and not isinstance(e.item, OpenAIResponseCompaction)
        ]
        assert all(i > done_index for i in other_output_item_indices)

    async def test_previous_response_id_chains_correctly_across_two_compacting_turns(
        self, openai_responses_impl, mock_inference_api, mock_responses_store
    ):
        """End-to-end chaining check tying together points 4 and the follow-up fix for
        the compaction item being double-counted in the next turn's history
        (TestCompactionItemNotDuplicatedInNextHistory): build on a response that
        itself compacted, and confirm the second turn still only summarizes its
        history once, not twice.
        """
        mock_responses_store.get_response_object.return_value = _previous_response()
        mock_inference_api.openai_chat_completion = _dual_inference_mock()

        first_result = await openai_responses_impl.create_openai_response(
            CreateResponseRequest(
                input="What is the code word?",
                model="meta-llama/Llama-3.1-8B-Instruct",
                previous_response_id="resp-previous-123",
                context_management=[{"type": "compaction", "compact_threshold": 1}],
            )
        )
        assert isinstance(first_result.output[0], OpenAIResponseCompaction)

        # Capture what the first (compacting) turn actually stored, and feed it back
        # as the second turn's previous_response -- mirroring how get_response_object
        # would resolve a real previous_response_id chained onto this one.
        final_store_call = next(
            call
            for call in mock_responses_store.upsert_response_object.call_args_list
            if call.kwargs["response_object"].status == "completed"
        )
        stored_input = final_store_call.kwargs["input"]
        chained_previous_response = _previous_response()
        chained_previous_response.input = stored_input
        chained_previous_response.output = first_result.output
        chained_previous_response.messages = final_store_call.kwargs["messages"]
        mock_responses_store.get_response_object.return_value = chained_previous_response

        await openai_responses_impl.create_openai_response(
            CreateResponseRequest(
                input="And the follow-up?",
                model="meta-llama/Llama-3.1-8B-Instruct",
                previous_response_id=first_result.id,
                context_management=[{"type": "compaction", "compact_threshold": 1}],
            )
        )

        second_compaction_call = [
            call for call in mock_inference_api.openai_chat_completion.call_args_list if not call.args[0].stream
        ][-1]
        compaction_item_count = sum(
            1
            for m in second_compaction_call.args[0].messages
            if getattr(m, "content", None) and "Summary of prior turns." in str(m.content)
        )
        assert compaction_item_count == 1
