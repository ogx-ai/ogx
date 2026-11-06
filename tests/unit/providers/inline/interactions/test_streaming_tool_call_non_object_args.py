# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""#6912: Google's function-call args must end up an object in streaming, the same way
the non-streaming path already makes it one (test_impl.py's TestToolCallingResponseTranslation),
without buffering each tool call's whole argument string to do it. Split from
test_impl.py to stay under the repo's per-file line limit.
"""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from ogx.providers.inline.interactions.config import InteractionsConfig
from ogx.providers.inline.interactions.impl import BuiltinInteractionsImpl


@pytest.fixture
def impl():
    mock_inference = AsyncMock()
    instance = BuiltinInteractionsImpl(config=InteractionsConfig(), inference_api=mock_inference, policy=[])
    instance.store = AsyncMock()
    instance.store.get_interaction = AsyncMock(return_value=None)
    instance.store.store_interaction = AsyncMock()
    return instance


def _make_tool_call_chunk(tc_idx, arguments, *, name=None, call_id=None, finish_reason=None):
    tc_delta = MagicMock()
    tc_delta.index = tc_idx
    tc_delta.id = call_id
    tc_delta.function = MagicMock()
    tc_delta.function.name = name
    tc_delta.function.arguments = arguments

    chunk = MagicMock()
    chunk.choices = [MagicMock()]
    chunk.choices[0].delta = MagicMock()
    chunk.choices[0].delta.content = None
    chunk.choices[0].delta.tool_calls = [tc_delta]
    chunk.choices[0].finish_reason = finish_reason
    chunk.usage = None
    return chunk


class TestToolCallingStreamingNonObjectArguments:
    @staticmethod
    async def _run(impl, chunks):
        async def stream():
            for c in chunks:
                yield c

        events = [event async for event in impl._stream_openai_to_google(stream(), "m", [])]
        arg_deltas = [e for e in events if e.event_type == "content.delta" and e.delta.type == "function_call"]
        return "".join(d.delta.args for d in arg_deltas)

    async def test_valid_object_split_across_chunks_streams_through_unchanged(self, impl):
        chunks = [
            _make_tool_call_chunk(0, '{"city":', name="get_weather", call_id="call_1"),
            _make_tool_call_chunk(0, ' "Boston"}', finish_reason="tool_calls"),
        ]
        joined = await self._run(impl, chunks)
        assert json.loads(joined) == {"city": "Boston"}

    @pytest.mark.parametrize("arguments", ["[1, 2]", '"hello"', "42", "true", "null"])
    async def test_non_object_value_becomes_empty_object(self, impl, arguments):
        chunks = [
            _make_tool_call_chunk(0, arguments, name="get_weather", call_id="call_1", finish_reason="tool_calls"),
        ]
        joined = await self._run(impl, chunks)
        assert joined == "{}"

    async def test_leading_whitespace_before_object_is_held_back_then_flushed(self, impl):
        chunks = [
            _make_tool_call_chunk(0, " ", name="get_weather", call_id="call_1"),
            _make_tool_call_chunk(0, '{"city": "Boston"}', finish_reason="tool_calls"),
        ]
        joined = await self._run(impl, chunks)
        assert json.loads(joined) == {"city": "Boston"}

    async def test_truly_empty_arguments_still_close_as_empty_object(self, impl):
        chunks = [
            _make_tool_call_chunk(0, None, name="get_time", call_id="call_1", finish_reason="tool_calls"),
        ]
        joined = await self._run(impl, chunks)
        assert joined == "{}"

    async def test_two_tool_calls_are_decided_independently(self, impl):
        chunks = [
            _make_tool_call_chunk(0, '{"a": 1}', name="tool_a", call_id="call_1"),
            _make_tool_call_chunk(1, "[1, 2]", name="tool_b", call_id="call_2", finish_reason="tool_calls"),
        ]

        async def stream():
            for c in chunks:
                yield c

        events = [event async for event in impl._stream_openai_to_google(stream(), "m", [])]
        arg_deltas = [e for e in events if e.event_type == "content.delta" and e.delta.type == "function_call"]
        by_index: dict[int, str] = {}
        for e in arg_deltas:
            by_index[e.index] = by_index.get(e.index, "") + e.delta.args

        assert json.loads(by_index[0]) == {"a": 1}
        assert by_index[1] == "{}"
