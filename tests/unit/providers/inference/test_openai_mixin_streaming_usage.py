# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Tests for OpenAIMixin.coalesce_streaming_usage behavior.

Some OpenAI-compatible providers (notably Gemini's endpoint) violate the OpenAI
spec by including usage in every streaming chunk rather than only in the final
empty-choices chunk. This causes callers that accumulate usage across chunks to
overcount tokens.

Categories:
  - Default behavior: coalesce_streaming_usage=False passes chunks through unchanged
  - Fixed behavior: coalesce_streaming_usage=True strips usage from content chunks,
    emits a single compliant final usage chunk
  - Edge cases: no usage in any chunk, first/last chunk handling

Specific tests:
  - test_default_does_not_strip_usage: content chunks retain usage when flag is False
  - test_fix_strips_usage_from_content_chunks: content chunks have usage=None when flag is True
  - test_fix_emits_final_usage_chunk: a single usage-only chunk is appended after content
  - test_fix_no_usage_no_extra_chunk: no extra chunk emitted when no chunk carries usage
  - test_fix_preserves_last_usage: the final chunk carries the last seen usage, not the first
  - test_gemini_adapter_has_flag_set: GeminiInferenceAdapter.coalesce_streaming_usage is True
"""

from collections.abc import AsyncIterator
from typing import Any

from openai.types import Completion, CompletionUsage
from openai.types.chat import ChatCompletionChunk
from openai.types.chat.chat_completion_chunk import Choice, ChoiceDelta
from openai.types.completion_choice import CompletionChoice


def _make_content_chunk(
    content: str,
    chunk_id: str = "cmp-1",
    model: str = "gemini-2.5-flash",
    usage: CompletionUsage | None = None,
) -> ChatCompletionChunk:
    return ChatCompletionChunk(
        id=chunk_id,
        choices=[Choice(delta=ChoiceDelta(content=content), finish_reason=None, index=0)],
        created=1000,
        model=model,
        object="chat.completion.chunk",
        usage=usage,
    )


def _make_stop_chunk(
    content: str = "",
    chunk_id: str = "cmp-1",
    model: str = "gemini-2.5-flash",
    usage: CompletionUsage | None = None,
) -> ChatCompletionChunk:
    return ChatCompletionChunk(
        id=chunk_id,
        choices=[Choice(delta=ChoiceDelta(content=content), finish_reason="stop", index=0)],
        created=1000,
        model=model,
        object="chat.completion.chunk",
        usage=usage,
    )


async def _collect(gen: AsyncIterator[Any]) -> list[Any]:
    return [item async for item in gen]


class TestDefaultBehavior:
    """coalesce_streaming_usage=False — chunks pass through unchanged."""

    async def test_default_does_not_strip_usage(self):
        """When coalesce_streaming_usage is False, chunks with usage are yielded as-is."""
        from ogx.providers.utils.inference.openai_mixin import OpenAIMixin

        usage = CompletionUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        chunks = [
            _make_content_chunk("hello", usage=usage),
            _make_content_chunk(" world", usage=usage),
        ]

        async def _fake_stream():
            for c in chunks:
                yield c

        # Instantiate a minimal stub that inherits _postprocess_chunk
        class _Stub(OpenAIMixin):
            config: Any = None
            coalesce_streaming_usage: bool = False
            overwrite_completion_id: bool = False

            def get_base_url(self):
                return "https://example.com"

        stub = _Stub.model_construct()
        stub.coalesce_streaming_usage = False
        stub.overwrite_completion_id = False

        result = await stub._postprocess_chunk(_fake_stream(), stream=True)
        collected = await _collect(result)

        assert len(collected) == 2
        for chunk in collected:
            assert chunk.usage is not None
            assert chunk.usage.completion_tokens == 5


class TestFixStreamingUsage:
    """coalesce_streaming_usage=True — strips usage from content chunks, appends final usage chunk."""

    def _make_stub(self) -> Any:
        from ogx.providers.utils.inference.openai_mixin import OpenAIMixin

        class _Stub(OpenAIMixin):
            config: Any = None
            coalesce_streaming_usage: bool = True
            overwrite_completion_id: bool = False

            def get_base_url(self):
                return "https://example.com"

        stub = _Stub.model_construct()
        stub.coalesce_streaming_usage = True
        stub.overwrite_completion_id = False
        return stub

    async def test_fix_strips_usage_from_content_chunks(self):
        """Content chunks have usage=None when coalesce_streaming_usage is True."""
        usage = CompletionUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        chunks = [
            _make_content_chunk("hello", usage=usage),
            _make_stop_chunk(usage=usage),
        ]

        async def _fake_stream():
            for c in chunks:
                yield c

        stub = self._make_stub()
        result = await stub._postprocess_chunk(_fake_stream(), stream=True)
        collected = await _collect(result)

        # 2 content/stop chunks + 1 final usage-only chunk
        assert len(collected) == 3
        # Original chunks have usage stripped
        assert collected[0].usage is None
        assert collected[1].usage is None

    async def test_fix_emits_final_usage_chunk(self):
        """A single usage-only chunk with empty choices is appended at the end."""
        usage = CompletionUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        chunks = [
            _make_content_chunk("hello", usage=usage),
            _make_stop_chunk(usage=usage),
        ]

        async def _fake_stream():
            for c in chunks:
                yield c

        stub = self._make_stub()
        result = await stub._postprocess_chunk(_fake_stream(), stream=True)
        collected = await _collect(result)

        final = collected[-1]
        assert final.choices == []
        assert final.usage is not None
        assert final.usage.prompt_tokens == 10
        assert final.usage.completion_tokens == 5
        assert final.usage.total_tokens == 15

    async def test_fix_no_usage_no_extra_chunk(self):
        """No extra chunk is emitted when no incoming chunk carries usage."""
        chunks = [
            _make_content_chunk("hello"),
            _make_stop_chunk(),
        ]

        async def _fake_stream():
            for c in chunks:
                yield c

        stub = self._make_stub()
        result = await stub._postprocess_chunk(_fake_stream(), stream=True)
        collected = await _collect(result)

        assert len(collected) == 2
        for chunk in collected:
            assert chunk.usage is None

    async def test_fix_preserves_last_usage(self):
        """The final chunk carries all token counts from the last seen usage chunk."""
        first_usage = CompletionUsage(prompt_tokens=10, completion_tokens=3, total_tokens=13)
        last_usage = CompletionUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        chunks = [
            _make_content_chunk("hello", usage=first_usage),
            _make_stop_chunk(usage=last_usage),
        ]

        async def _fake_stream():
            for c in chunks:
                yield c

        stub = self._make_stub()
        result = await stub._postprocess_chunk(_fake_stream(), stream=True)
        collected = await _collect(result)

        final = collected[-1]
        # All three counts come from last_usage, not first_usage
        assert final.usage.prompt_tokens == 10
        assert final.usage.completion_tokens == 5
        assert final.usage.total_tokens == 15

    async def test_fix_gemini_cumulative_usage_pattern(self):
        """Gemini reports cumulative usage on every chunk; only the last gives correct totals.

        Gemini's buggy endpoint sends usage on every chunk. completion_tokens and
        total_tokens are running totals that grow with each chunk; prompt_tokens stays
        constant. Real example (gemini-2.5-flash-lite):

            chunk 1: completion_tokens=3,  prompt_tokens=9, total_tokens=12
            chunk 2: completion_tokens=19, prompt_tokens=9, total_tokens=28
            chunk 3: completion_tokens=63, prompt_tokens=9, total_tokens=72  ← correct

        Taking only the last chunk's usage is the correct strategy — prompt_tokens and
        total_tokens must NOT be summed across chunks.
        """
        chunks = [
            _make_content_chunk(
                "Streaming APIs allow",
                usage=CompletionUsage(prompt_tokens=9, completion_tokens=3, total_tokens=12),
            ),
            _make_content_chunk(
                " for the continuous, real-time delivery of data as it's generated,",
                usage=CompletionUsage(prompt_tokens=9, completion_tokens=19, total_tokens=28),
            ),
            _make_stop_chunk(
                " rather than requiring a client to poll for updates or download large datasets all at once. This is achieved through persistent connections that push new information to subscribers as soon as it becomes available, enabling applications to react instantly to changes.",
                usage=CompletionUsage(prompt_tokens=9, completion_tokens=63, total_tokens=72),
            ),
        ]

        async def _fake_stream():
            for c in chunks:
                yield c

        stub = self._make_stub()
        result = await stub._postprocess_chunk(_fake_stream(), stream=True)
        collected = await _collect(result)

        # 3 content/stop chunks + 1 final usage-only chunk
        assert len(collected) == 4
        # No usage on the yielded content chunks
        assert all(c.usage is None for c in collected[:3])

        final = collected[-1]
        assert final.choices == []
        # Must reflect the last chunk's values, not a sum across chunks
        assert final.usage.prompt_tokens == 9  # not 27 (9*3)
        assert final.usage.completion_tokens == 63  # not 85 (3+19+63)
        assert final.usage.total_tokens == 72  # not 112 (12+28+72)

    async def test_fix_single_chunk_with_usage(self):
        """Works correctly when there is only one chunk and it carries usage."""
        usage = CompletionUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        chunks = [_make_stop_chunk(usage=usage)]

        async def _fake_stream():
            for c in chunks:
                yield c

        stub = self._make_stub()
        result = await stub._postprocess_chunk(_fake_stream(), stream=True)
        collected = await _collect(result)

        assert len(collected) == 2
        assert collected[0].usage is None
        assert collected[1].choices == []
        assert collected[1].usage.total_tokens == 15

    async def test_fix_preserves_chunk_id_and_model_on_final(self):
        """The final usage chunk carries the correct id and model from the last content chunk."""
        usage = CompletionUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        chunks = [_make_stop_chunk(chunk_id="my-id", model="gemini-2.5-pro", usage=usage)]

        async def _fake_stream():
            for c in chunks:
                yield c

        stub = self._make_stub()
        result = await stub._postprocess_chunk(_fake_stream(), stream=True)
        collected = await _collect(result)

        final = collected[-1]
        assert final.id == "my-id"
        assert final.model == "gemini-2.5-pro"


def _make_text_completion_chunk(
    text: str,
    chunk_id: str = "cmpl-1",
    model: str = "gemini-2.5-flash",
    finish_reason: str | None = None,
    usage: CompletionUsage | None = None,
) -> Completion:
    # Intermediate stream chunks have finish_reason=None; the SDK parses stream
    # events leniently (construct), so mirror that here.
    choice = CompletionChoice.model_construct(text=text, finish_reason=finish_reason, index=0, logprobs=None)
    return Completion(
        id=chunk_id,
        choices=[choice],
        created=1000,
        model=model,
        object="text_completion",
        usage=usage,
    )


class TestFixStreamingUsageTextCompletion:
    """coalesce_streaming_usage=True on the legacy /v1/completions stream.

    The final usage-only chunk must be a Completion (object='text_completion'),
    not a ChatCompletionChunk — OpenAI SDK clients parse every event of a
    text-completion stream as Completion and fail validation otherwise.
    """

    def _make_stub(self) -> Any:
        from ogx.providers.utils.inference.openai_mixin import OpenAIMixin

        class _Stub(OpenAIMixin):
            config: Any = None
            coalesce_streaming_usage: bool = True
            overwrite_completion_id: bool = False

            def get_base_url(self):
                return "https://example.com"

        stub = _Stub.model_construct()
        stub.coalesce_streaming_usage = True
        stub.overwrite_completion_id = False
        return stub

    async def test_fix_completion_stream_yields_only_text_completion_chunks(self):
        """Every yielded chunk, including the final usage chunk, is a text_completion."""
        usage = CompletionUsage(prompt_tokens=9, completion_tokens=5, total_tokens=14)
        chunks = [
            _make_text_completion_chunk("hello", usage=usage),
            _make_text_completion_chunk(" world", finish_reason="stop", usage=usage),
        ]

        async def _fake_stream():
            for c in chunks:
                yield c

        stub = self._make_stub()
        result = await stub._postprocess_chunk(_fake_stream(), stream=True, is_chat=False)
        collected = await _collect(result)

        assert len(collected) == 3
        for chunk in collected:
            assert isinstance(chunk, Completion), f"got {type(chunk).__name__}"
            assert chunk.object == "text_completion"

    async def test_fix_completion_stream_final_chunk_carries_usage(self):
        """Usage is stripped from content chunks and delivered on a final Completion chunk."""
        usage = CompletionUsage(prompt_tokens=9, completion_tokens=5, total_tokens=14)
        chunks = [
            _make_text_completion_chunk("hello", usage=usage),
            _make_text_completion_chunk(" world", finish_reason="stop", usage=usage),
        ]

        async def _fake_stream():
            for c in chunks:
                yield c

        stub = self._make_stub()
        result = await stub._postprocess_chunk(_fake_stream(), stream=True, is_chat=False)
        collected = await _collect(result)

        assert all(c.usage is None for c in collected[:-1])
        final = collected[-1]
        assert final.choices == []
        assert final.id == "cmpl-1"
        assert final.model == "gemini-2.5-flash"
        assert final.usage is not None
        assert final.usage.prompt_tokens == 9
        assert final.usage.completion_tokens == 5
        assert final.usage.total_tokens == 14

    async def test_fix_completion_stream_no_usage_no_extra_chunk(self):
        """No synthetic chunk is appended when no incoming chunk carries usage."""
        chunks = [
            _make_text_completion_chunk("hello"),
            _make_text_completion_chunk(" world", finish_reason="stop"),
        ]

        async def _fake_stream():
            for c in chunks:
                yield c

        stub = self._make_stub()
        result = await stub._postprocess_chunk(_fake_stream(), stream=True, is_chat=False)
        collected = await _collect(result)

        assert len(collected) == 2
        assert all(isinstance(c, Completion) for c in collected)

    async def test_chat_stream_final_chunk_still_chat_completion_chunk(self):
        """The chat path is unchanged: final usage chunk stays a ChatCompletionChunk."""
        usage = CompletionUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        chunks = [_make_stop_chunk(usage=usage)]

        async def _fake_stream():
            for c in chunks:
                yield c

        stub = self._make_stub()
        result = await stub._postprocess_chunk(_fake_stream(), stream=True)
        collected = await _collect(result)

        final = collected[-1]
        assert isinstance(final, ChatCompletionChunk)
        assert final.object == "chat.completion.chunk"
        assert final.usage.total_tokens == 15


class TestGeminiAdapterFlag:
    def test_gemini_adapter_has_coalesce_streaming_usage_enabled(self):
        """GeminiInferenceAdapter must have coalesce_streaming_usage=True."""
        from ogx.providers.remote.inference.gemini.gemini import GeminiInferenceAdapter

        # Pydantic intercepts class-attribute access on model fields, so read the
        # default from model_fields instead.
        default = GeminiInferenceAdapter.model_fields["coalesce_streaming_usage"].default
        assert default is True
