# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Automatic compaction (context_management) via previous_response_id and single-call
input: the new turn always follows the compaction item, no hidden response is stored,
instructions/extra_body reach the summarizer, the summary has its own output budget and
its usage is reported, the threshold counts current context, and a failed or empty
summary falls back to the original history (#6829).
"""

from unittest.mock import AsyncMock

from ogx.providers.inline.responses.builtin.responses.openai_responses import OpenAIResponsesImpl
from ogx_api.openai_responses import (
    OpenAIResponseCompaction,
    OpenAIResponseMessage,
)
from ogx_api.responses.models import CreateResponseRequest
from tests.unit.providers.responses.builtin.compaction_test_helpers import (
    _compaction_completion,
    _dual_inference_mock,
    _previous_response,
)
from tests.unit.providers.responses.builtin.test_openai_responses_helpers import fake_stream


class TestMaybeAutoCompactOrdering:
    """Point 1: the new input must always end up after the compaction item, never
    before it -- a compaction item renders as an assistant message
    (utils.py's compaction -> OpenAIAssistantMessageParam conversion), so putting the
    new question before it makes the model's own summary its last message.
    """

    async def _make_impl(self, mock_inference_api):
        impl = object.__new__(OpenAIResponsesImpl)
        impl.inference_api = mock_inference_api
        impl.files_api = AsyncMock()
        from ogx.providers.inline.responses.builtin.config import CompactionConfig

        impl.compaction_config = CompactionConfig()
        return impl

    async def test_new_input_follows_the_compaction_item(self, mock_inference_api):
        impl = await self._make_impl(mock_inference_api)
        mock_inference_api.openai_chat_completion = _dual_inference_mock()

        history = [
            OpenAIResponseMessage(role="user", content="Turn one."),
            OpenAIResponseMessage(role="assistant", content="Reply one."),
        ]
        new_input = [OpenAIResponseMessage(role="user", content="What is the code word?")]

        effective_input, compaction_item, chat_usage = await impl._maybe_auto_compact(
            history, new_input, "test-model", [{"type": "compaction", "compact_threshold": 1}]
        )

        assert compaction_item is not None
        assert chat_usage is not None
        assert effective_input[-1] is new_input[0]
        assert isinstance(effective_input[-2], OpenAIResponseCompaction)

    async def test_below_threshold_leaves_history_and_new_input_untouched(self, mock_inference_api):
        impl = await self._make_impl(mock_inference_api)
        mock_inference_api.openai_chat_completion = _dual_inference_mock()

        history = [OpenAIResponseMessage(role="user", content="hi")]
        new_input = [OpenAIResponseMessage(role="user", content="again")]

        effective_input, compaction_item, chat_usage = await impl._maybe_auto_compact(
            history, new_input, "test-model", [{"type": "compaction", "compact_threshold": 100000}]
        )

        assert compaction_item is None
        assert chat_usage is None
        assert effective_input == history + new_input
        mock_inference_api.openai_chat_completion.assert_not_called()

    async def test_empty_history_short_circuits_without_calling_inference(self, mock_inference_api):
        impl = await self._make_impl(mock_inference_api)
        mock_inference_api.openai_chat_completion = _dual_inference_mock()

        new_input = [OpenAIResponseMessage(role="user", content="hello")]
        effective_input, compaction_item, chat_usage = await impl._maybe_auto_compact(
            [], new_input, "test-model", [{"type": "compaction", "compact_threshold": 1}]
        )

        assert compaction_item is None
        assert effective_input == new_input
        mock_inference_api.openai_chat_completion.assert_not_called()

    async def test_empty_summary_keeps_original_history(self, mock_inference_api):
        """Point 6: a reasoning model hitting max_output_tokens mid-thought can return
        content=None. Must not silently replace history with a hollow/empty summary.
        """
        impl = await self._make_impl(mock_inference_api)
        mock_inference_api.openai_chat_completion = _dual_inference_mock(summary=None)

        history = [OpenAIResponseMessage(role="user", content="hi")]
        new_input = [OpenAIResponseMessage(role="user", content="again")]

        effective_input, compaction_item, chat_usage = await impl._maybe_auto_compact(
            history, new_input, "test-model", [{"type": "compaction", "compact_threshold": 1}]
        )

        assert compaction_item is None
        assert chat_usage is None
        assert effective_input == history + new_input

    async def test_truncated_summary_is_kept_but_logs_a_warning(self, mock_inference_api, caplog):
        """Code review of #6838: a non-empty summary cut off by its own output-token
        cap (finish_reason == "length") is still usable context, so it isn't failed the
        way an empty summary is -- but admins tuning
        CompactionConfig.summary_max_output_tokens need to be able to see it happened.
        """
        impl = await self._make_impl(mock_inference_api)

        async def side_effect(params):
            if params.stream:
                return fake_stream()
            return _compaction_completion("Truncated sum", finish_reason="length")

        mock_inference_api.openai_chat_completion = AsyncMock(side_effect=side_effect)

        history = [
            OpenAIResponseMessage(role="user", content="Turn one."),
            OpenAIResponseMessage(role="assistant", content="Reply one."),
        ]
        new_input = [OpenAIResponseMessage(role="user", content="What is the code word?")]

        with caplog.at_level("WARNING"):
            _, compaction_item, chat_usage = await impl._maybe_auto_compact(
                history, new_input, "test-model", [{"type": "compaction", "compact_threshold": 1}]
            )

        assert compaction_item is not None
        assert chat_usage is not None
        assert compaction_item.encrypted_content.endswith("Truncated sum")
        assert "truncated" in caplog.text.lower()


class TestAutoCompactionEndToEnd:
    """End-to-end tests driving the real create_openai_response path (not mocking
    _maybe_auto_compact), so these exercise the full wiring: call-site splitting,
    orchestrator seeding, and storage.
    """

    async def test_no_hidden_response_is_ever_stored(
        self, openai_responses_impl, mock_inference_api, mock_responses_store
    ):
        """Points 2 & 3: the automatic path must never create the kind of orphaned,
        store-ignoring response compact_openai_response creates for the manual endpoint.
        """
        mock_responses_store.get_response_object.return_value = _previous_response()
        mock_inference_api.openai_chat_completion = _dual_inference_mock()

        await openai_responses_impl.create_openai_response(
            CreateResponseRequest(
                input="What is the code word?",
                model="meta-llama/Llama-3.1-8B-Instruct",
                previous_response_id="resp-previous-123",
                context_management=[{"type": "compaction", "compact_threshold": 1}],
                store=False,
            )
        )

        mock_responses_store.store_response_object.assert_not_called()

    async def test_instructions_and_extra_body_reach_the_summarization_call(
        self, openai_responses_impl, mock_inference_api, mock_responses_store
    ):
        """Point 3 (plus the extra_body point): these were all silently dropped by the
        automatic path before this fix.
        """
        mock_responses_store.get_response_object.return_value = _previous_response()
        mock_inference_api.openai_chat_completion = _dual_inference_mock()

        await openai_responses_impl.create_openai_response(
            CreateResponseRequest(
                input="What is the code word?",
                model="meta-llama/Llama-3.1-8B-Instruct",
                previous_response_id="resp-previous-123",
                instructions="Be terse.",
                context_management=[{"type": "compaction", "compact_threshold": 1}],
                extra_body={"cache_salt": "abc123"},
            )
        )

        compaction_call = next(
            call for call in mock_inference_api.openai_chat_completion.call_args_list if not call.args[0].stream
        )
        params = compaction_call.args[0]
        assert params.cache_salt == "abc123"
        assert any("Be terse." in str(m.content) for m in params.messages)

    async def test_the_answers_max_output_tokens_never_caps_the_summarization_call(
        self, openai_responses_impl, mock_inference_api, mock_responses_store
    ):
        """Review of #6829: the summarization call was capped by the *answer's own*
        max_output_tokens (openai_responses.py's old _maybe_auto_compact signature took
        it straight from request.max_output_tokens), so a short max_output_tokens could
        let the summary alone exhaust it, leaving nothing for the actual answer. The
        summary must get its own independent cap (CompactionConfig.summary_max_output_tokens).
        """
        mock_responses_store.get_response_object.return_value = _previous_response()
        mock_inference_api.openai_chat_completion = _dual_inference_mock()

        await openai_responses_impl.create_openai_response(
            CreateResponseRequest(
                input="What is the code word?",
                model="meta-llama/Llama-3.1-8B-Instruct",
                previous_response_id="resp-previous-123",
                max_output_tokens=50,
                context_management=[{"type": "compaction", "compact_threshold": 1}],
            )
        )

        compaction_call = next(
            call for call in mock_inference_api.openai_chat_completion.call_args_list if not call.args[0].stream
        )
        assert (
            compaction_call.args[0].max_completion_tokens
            == openai_responses_impl.compaction_config.summary_max_output_tokens
        )
        assert compaction_call.args[0].max_completion_tokens != 50

    async def test_summarization_usage_does_not_reduce_the_answers_own_output_budget(
        self, openai_responses_impl, mock_inference_api, mock_responses_store
    ):
        """The summarization call's completion tokens must still show up in the final
        usage (test_compaction_usage_is_folded_into_the_final_response_usage covers
        that), but must not count against max_output_tokens' budget for the main answer
        -- otherwise a response can come back incomplete with only the compaction item
        and no answer, purely because the summary itself used up the whole budget.
        """
        mock_responses_store.get_response_object.return_value = _previous_response()
        # A summary that alone would exhaust a small max_output_tokens if it counted
        # against the answer's budget.
        mock_inference_api.openai_chat_completion = _dual_inference_mock(completion_tokens=40)

        result = await openai_responses_impl.create_openai_response(
            CreateResponseRequest(
                input="What is the code word?",
                model="meta-llama/Llama-3.1-8B-Instruct",
                previous_response_id="resp-previous-123",
                max_output_tokens=50,
                context_management=[{"type": "compaction", "compact_threshold": 1}],
            )
        )

        assert result.status == "completed"
        assert result.incomplete_details is None
        main_call = next(
            call for call in mock_inference_api.openai_chat_completion.call_args_list if call.args[0].stream
        )
        # The main call must still see the full max_output_tokens budget, not
        # max_output_tokens minus the summary's own completion tokens.
        assert main_call.args[0].max_completion_tokens == 50

    async def test_compaction_usage_is_folded_into_the_final_response_usage(
        self, openai_responses_impl, mock_inference_api, mock_responses_store
    ):
        """Point 4: the summarization call's tokens must show up somewhere, not be
        silently dropped while still being billed by the provider.
        """
        mock_responses_store.get_response_object.return_value = _previous_response()
        mock_inference_api.openai_chat_completion = _dual_inference_mock(prompt_tokens=100, completion_tokens=50)

        result = await openai_responses_impl.create_openai_response(
            CreateResponseRequest(
                input="What is the code word?",
                model="meta-llama/Llama-3.1-8B-Instruct",
                previous_response_id="resp-previous-123",
                context_management=[{"type": "compaction", "compact_threshold": 1}],
            )
        )

        assert result.usage is not None
        # 150 from the compaction call, plus whatever the main turn's fake_stream usage is.
        assert result.usage.input_tokens >= 100
        assert result.usage.output_tokens >= 50

    async def test_threshold_checks_current_context_not_stale_previous_usage(
        self, openai_responses_impl, mock_inference_api, mock_responses_store
    ):
        """Point 5: previous_usage.total_tokens reflects the turn *before* the new
        question existed. A request that's only over threshold once the new question is
        counted must still trigger compaction.
        """
        # previous_usage.total_tokens (10) is comfortably under the threshold (50); only
        # counting the actual current text (a long history + new question) exceeds it.
        long_text = "word " * 100
        previous_response = _previous_response(input_text=long_text, output_text=long_text)
        mock_responses_store.get_response_object.return_value = previous_response
        mock_inference_api.openai_chat_completion = _dual_inference_mock()

        await openai_responses_impl.create_openai_response(
            CreateResponseRequest(
                input="What is the code word?",
                model="meta-llama/Llama-3.1-8B-Instruct",
                previous_response_id="resp-previous-123",
                context_management=[{"type": "compaction", "compact_threshold": 50}],
            )
        )

        compaction_calls = [
            call for call in mock_inference_api.openai_chat_completion.call_args_list if not call.args[0].stream
        ]
        assert len(compaction_calls) == 1

    async def test_fresh_single_call_with_large_input_still_compacts(
        self, openai_responses_impl, mock_inference_api, mock_responses_store
    ):
        """No previous_response_id or conversation at all -- a single oversized first
        call must still split its own trailing turn off and compact everything before
        it (mirrors the integration test test_context_management_auto_compacts_large_input).
        """
        mock_inference_api.openai_chat_completion = _dual_inference_mock()

        large_input = []
        for i in range(10):
            large_input.append(OpenAIResponseMessage(role="user", content=f"Tell me about topic {i}."))
            large_input.append(OpenAIResponseMessage(role="assistant", content=f"Here is topic {i}. " * 20))
        large_input.append(OpenAIResponseMessage(role="user", content="Summarize what we discussed."))

        await openai_responses_impl.create_openai_response(
            CreateResponseRequest(
                input=large_input,
                model="meta-llama/Llama-3.1-8B-Instruct",
                context_management=[{"type": "compaction", "compact_threshold": 50}],
            )
        )

        compaction_calls = [
            call for call in mock_inference_api.openai_chat_completion.call_args_list if not call.args[0].stream
        ]
        assert len(compaction_calls) == 1
        # The final question must still be the last message sent to the model for the
        # main turn, not swallowed into the compacted summary.
        main_call = next(
            call for call in mock_inference_api.openai_chat_completion.call_args_list if call.args[0].stream
        )
        assert main_call.args[0].messages[-1].content == "Summarize what we discussed."


class TestPreviousResponseIdChaining:
    """Multi-turn chaining check requested in code review of #6838's Test Plan
    section, exercising a second previous_response_id turn built on a compacting
    one -- not just the final response object a single call produces.
    """

    async def test_previous_response_id_chains_correctly_across_two_compacting_turns(
        self, openai_responses_impl, mock_inference_api, mock_responses_store
    ):
        """End-to-end chaining check tying together point 4: build on a response that
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
