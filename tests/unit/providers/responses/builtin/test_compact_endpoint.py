# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""The manual /v1/responses/compact endpoint: how request fields reach the
summarization call, its output-token cap, and its empty-summary handling, which
deliberately differs from the automatic path (the caller explicitly asked to compact,
so a hollow result is surfaced as an error rather than silently kept as-is).
"""

from unittest.mock import AsyncMock

import pytest

from ogx.providers.inline.responses.builtin.impl import BuiltinResponsesImpl
from ogx_api import InternalServerError, InvalidParameterError
from ogx_api.responses.models import CompactResponseRequest
from tests.unit.providers.responses.builtin.compaction_test_helpers import _dual_inference_mock


def _wrapper() -> BuiltinResponsesImpl:
    wrapper = object.__new__(BuiltinResponsesImpl)
    wrapper.openai_responses_impl = AsyncMock()
    return wrapper


class TestCompactRequestForwarding:
    """CompactResponseRequest has no typed extra_body field, only extra="allow". Every
    top-level extra field must keep reaching the summarization call (vendor params such
    as top_k), with only names that collide with the call's own arguments dropped.
    """

    async def test_top_level_vendor_params_are_forwarded(self):
        wrapper = _wrapper()

        await wrapper.compact_openai_response(CompactResponseRequest(model="test-model", input="hi", top_k=5))

        call_kwargs = wrapper.openai_responses_impl.compact_openai_response.call_args.kwargs
        assert call_kwargs["extra_body"] == {"top_k": 5}
        assert call_kwargs["max_output_tokens"] is None

    async def test_nested_extra_body_is_merged_and_max_output_tokens_is_read_separately(self):
        wrapper = _wrapper()

        await wrapper.compact_openai_response(
            CompactResponseRequest(
                model="test-model",
                input="hi",
                top_k=5,
                max_output_tokens=77,
                extra_body={"cache_salt": "xyz"},
            )
        )

        call_kwargs = wrapper.openai_responses_impl.compact_openai_response.call_args.kwargs
        assert call_kwargs["extra_body"] == {"top_k": 5, "cache_salt": "xyz"}
        assert call_kwargs["max_output_tokens"] == 77

    async def test_no_extra_fields_forwards_none(self):
        wrapper = _wrapper()

        await wrapper.compact_openai_response(CompactResponseRequest(model="test-model", input="hi"))

        call_kwargs = wrapper.openai_responses_impl.compact_openai_response.call_args.kwargs
        assert call_kwargs["extra_body"] is None

    async def test_colliding_names_are_dropped_from_the_summarization_call(
        self, openai_responses_impl, mock_inference_api
    ):
        """A field named like one of the summarization call's own arguments (e.g. a
        client copy-pasting stream or max_completion_tokens from chat completions) must
        neither raise a duplicate-keyword TypeError nor override the call's settings.
        """
        mock_inference_api.openai_chat_completion = _dual_inference_mock()

        await openai_responses_impl.compact_openai_response(
            model="test-model",
            input="Some content to compact.",
            extra_body={"stream": True, "max_completion_tokens": 999, "model": "other", "top_k": 5},
        )

        params = mock_inference_api.openai_chat_completion.call_args.args[0]
        assert params.stream is False
        assert params.max_completion_tokens is None
        assert params.model == "test-model"
        assert params.top_k == 5


class TestManualCompactEndpoint:
    async def test_extra_body_and_max_output_tokens_reach_the_summarization_call(
        self, openai_responses_impl, mock_inference_api
    ):
        mock_inference_api.openai_chat_completion = _dual_inference_mock()

        await openai_responses_impl.compact_openai_response(
            model="test-model",
            input="Some content to compact.",
            extra_body={"cache_salt": "xyz"},
            max_output_tokens=77,
        )

        params = mock_inference_api.openai_chat_completion.call_args.args[0]
        assert params.max_completion_tokens == 77
        assert params.cache_salt == "xyz"

    async def test_max_output_tokens_defaults_to_the_configured_summary_cap(
        self, openai_responses_impl, mock_inference_api
    ):
        openai_responses_impl.compaction_config.summary_max_output_tokens = 777
        mock_inference_api.openai_chat_completion = _dual_inference_mock()

        await openai_responses_impl.compact_openai_response(model="test-model", input="Some content.")

        params = mock_inference_api.openai_chat_completion.call_args.args[0]
        assert params.max_completion_tokens == 777

    @pytest.mark.parametrize("bad_value", ["abc", 0, -5])
    async def test_bad_max_output_tokens_raises_invalid_parameter_error_named_after_itself(
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

    async def test_empty_summary_raises_a_server_error_not_a_bad_request(
        self, openai_responses_impl, mock_inference_api
    ):
        """An empty summary is the summarization model's failure, not a malformed
        request: ValueError maps to 400 (core/exceptions/mapping.py), InternalServerError
        to 500.
        """
        mock_inference_api.openai_chat_completion = _dual_inference_mock(summary=None)

        with pytest.raises(InternalServerError, match="empty summary"):
            await openai_responses_impl.compact_openai_response(model="test-model", input="Some content.")
