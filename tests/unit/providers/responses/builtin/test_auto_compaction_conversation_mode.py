# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Automatic compaction in conversation= mode: the whole conversation is compacted,
re-compaction reads the bounded stored message log rather than the ever-growing raw
item log, the raw item log is only fetched when needed, deleted conversations still
fail fast, and verbatim user messages and tool-call pairing survive compaction.
"""

from unittest.mock import AsyncMock

import pytest

from ogx_api import (
    ConversationNotFoundError,
    OpenAIUserMessageParam,
)
from ogx_api.openai_responses import (
    OpenAIResponseInputFunctionToolCallOutput,
)
from ogx_api.responses.models import CreateResponseRequest
from tests.unit.providers.responses.builtin.compaction_test_helpers import (
    _conversation_pages,
    _dual_inference_mock,
)
from tests.unit.providers.responses.builtin.test_openai_responses_helpers import fake_stream


class TestConversationModeCompaction:
    """Point 8: conversation= mode must compact the whole conversation, not just the
    newly submitted turn.
    """

    async def test_full_history_is_compacted_not_just_the_new_turn(
        self, openai_responses_impl, mock_inference_api, mock_responses_store, mock_conversations_api
    ):
        from ogx_api import OpenAIUserMessageParam

        mock_responses_store.get_conversation_messages.return_value = [
            OpenAIUserMessageParam(content=f"Message {i}") for i in range(25)
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

        # Performance note (code review of #6838): once a stored message log exists,
        # the conversation's raw item log is never fetched at all -- not even a single
        # page -- see TestConversationRawItemLogFetchedOnlyWhenNeeded.
        mock_conversations_api.list_items.assert_not_called()
        compaction_call = next(
            call for call in mock_inference_api.openai_chat_completion.call_args_list if not call.args[0].stream
        )
        # The summarization call's messages must reflect the full 25-item history, not
        # just the single new turn.
        assert len(compaction_call.args[0].messages) > 2

    async def test_non_compacting_conversation_turn_still_stores_only_the_new_turn(
        self, openai_responses_impl, mock_inference_api, mock_responses_store, mock_conversations_api
    ):
        """A turn where compaction doesn't fire must keep conversation= mode's existing
        O(1)-per-response storage behavior -- this fix must not make every conversation
        turn snapshot the full history.
        """
        from ogx_api import OpenAIUserMessageParam

        pages = _conversation_pages(total_items=5, page_size=10)
        mock_conversations_api.list_items = AsyncMock(side_effect=pages)
        mock_responses_store.get_conversation_messages.return_value = [
            OpenAIUserMessageParam(content=f"Message {i}") for i in range(5)
        ]
        mock_inference_api.openai_chat_completion = _dual_inference_mock()

        await openai_responses_impl.create_openai_response(
            CreateResponseRequest(
                input="A new question.",
                model="meta-llama/Llama-3.1-8B-Instruct",
                conversation="conv_272fa1af1b67b662ee7ef05f72ec300d92425257645a531a",
                context_management=[{"type": "compaction", "compact_threshold": 100000}],
            )
        )

        store_call_args = mock_responses_store.upsert_response_object.call_args
        stored_input = store_call_args.kwargs["input"]
        assert len(stored_input) == 1
        assert stored_input[0].content[0].text == "A new question."


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
        pages = _conversation_pages(total_items=25, page_size=10)
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

    async def test_recompaction_uses_the_stored_message_log_not_the_ever_growing_raw_items(
        self, openai_responses_impl, mock_inference_api, mock_responses_store, mock_conversations_api
    ):
        from ogx_api import OpenAIAssistantMessageParam

        # The raw conversation item log has already grown large, as if several turns
        # happened after an earlier compaction -- since compaction items are never
        # synced into it, it keeps every raw item forever and requires pagination.
        pages = _conversation_pages(total_items=150, page_size=100)
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
