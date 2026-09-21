# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Unit tests for stream_utils: stream wrapping and best-effort closing."""

from unittest.mock import MagicMock

import pytest

from ogx.providers.utils.inference.stream_utils import (
    close_async_stream,
    wrap_async_stream,
    wrap_reasoning_chunks,
)


class TrackingAcloseStream:
    """Async iterator that records whether aclose() was called."""

    def __init__(self, chunks, fail_at=None):
        self._chunks = list(chunks)
        self._fail_at = fail_at
        self._pos = 0
        self.closed = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._fail_at is not None and self._pos >= self._fail_at:
            raise RuntimeError("upstream failed")
        if self._pos >= len(self._chunks):
            raise StopAsyncIteration
        item = self._chunks[self._pos]
        self._pos += 1
        return item

    async def aclose(self):
        self.closed = True


class CloseOnlyStream:
    """Simulates the OpenAI SDK AsyncStream: exposes close(), not aclose()."""

    def __init__(self, chunks):
        self._chunks = list(chunks)
        self._pos = 0
        self.closed = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._pos >= len(self._chunks):
            raise StopAsyncIteration
        item = self._chunks[self._pos]
        self._pos += 1
        return item

    async def close(self):
        self.closed = True


class TestWrapAsyncStream:
    async def test_exhaustion_closes_stream(self):
        stream = TrackingAcloseStream([1, 2, 3])
        items = [item async for item in wrap_async_stream(stream)]
        assert items == [1, 2, 3]
        assert stream.closed

    async def test_early_consumer_close_closes_stream(self):
        stream = TrackingAcloseStream([1, 2, 3])
        wrapper = wrap_async_stream(stream)
        assert await wrapper.__anext__() == 1
        await wrapper.aclose()
        assert stream.closed

    async def test_upstream_error_closes_stream(self):
        stream = TrackingAcloseStream([1, 2, 3], fail_at=1)
        wrapper = wrap_async_stream(stream)
        assert await wrapper.__anext__() == 1
        with pytest.raises(RuntimeError, match="upstream failed"):
            await wrapper.__anext__()
        assert stream.closed

    async def test_close_only_stream_is_closed(self):
        stream = CloseOnlyStream([1, 2, 3])
        items = [item async for item in wrap_async_stream(stream)]
        assert items == [1, 2, 3]
        assert stream.closed


class TestCloseAsyncStream:
    async def test_closes_stream_with_aclose(self):
        stream = TrackingAcloseStream([1])
        await close_async_stream(stream)
        assert stream.closed

    async def test_handles_sync_close(self):
        class SyncCloseStream:
            def close(self):
                self.closed = True

        stream = SyncCloseStream()
        await close_async_stream(stream)
        assert stream.closed

    async def test_close_error_is_suppressed(self):
        class FailingCloseStream:
            async def aclose(self):
                raise RuntimeError("close failed")

        await close_async_stream(FailingCloseStream())

    async def test_missing_close_is_noop(self):
        await close_async_stream(object())


def _chunk_with_reasoning(reasoning=None, reasoning_content=None):
    chunk = MagicMock()
    chunk.choices = [MagicMock()]
    chunk.choices[0].delta = MagicMock()
    chunk.choices[0].delta.reasoning = reasoning
    chunk.choices[0].delta.reasoning_content = reasoning_content
    return chunk


class TestWrapReasoningChunks:
    async def test_extracts_reasoning_from_delta(self):
        chunks = [
            _chunk_with_reasoning(reasoning="step 1"),
            _chunk_with_reasoning(reasoning_content="step 2"),
            _chunk_with_reasoning(),
        ]
        stream = TrackingAcloseStream(chunks)
        items = [item async for item in wrap_reasoning_chunks(stream)]
        assert [item.reasoning_content for item in items] == ["step 1", "step 2", None]
        assert stream.closed

    async def test_early_consumer_close_closes_stream(self):
        stream = TrackingAcloseStream([_chunk_with_reasoning(reasoning="x")] * 3)
        wrapper = wrap_reasoning_chunks(stream)
        assert (await wrapper.__anext__()).reasoning_content == "x"
        await wrapper.aclose()
        assert stream.closed

    async def test_upstream_error_closes_stream(self):
        stream = TrackingAcloseStream([_chunk_with_reasoning(reasoning="x")], fail_at=1)
        wrapper = wrap_reasoning_chunks(stream)
        await wrapper.__anext__()
        with pytest.raises(RuntimeError, match="upstream failed"):
            await wrapper.__anext__()
        assert stream.closed
