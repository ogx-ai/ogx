# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""
Regression tests for issue #3185: AsyncStream passed where AsyncIterator expected.

The bug: OpenAI SDK's AsyncStream has close(), not aclose(), but Python's
AsyncIterator protocol requires aclose(). The fix ensures _maybe_overwrite_id()
always wraps streaming responses in an async generator.
"""

import inspect
from collections.abc import AsyncIterator
from unittest.mock import MagicMock

import pytest

from ogx.providers.utils.inference.model_registry import RemoteInferenceProviderConfig
from ogx.providers.utils.inference.openai_mixin import OpenAIMixin


class MockAsyncStream:
    """Simulates OpenAI SDK's AsyncStream: has close() but NOT aclose()."""

    def __init__(self, chunks):
        self.chunks = chunks
        self._iter = iter(chunks)

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self._iter)
        except StopIteration as e:
            raise StopAsyncIteration from e

    async def close(self):
        pass


class MockChunk:
    def __init__(self, chunk_id: str, content: str = "test"):
        self.id = chunk_id
        self.content = content


class OpenAIMixinTestImpl(OpenAIMixin):
    __provider_id__: str = "test-provider"

    def get_api_key(self) -> str:
        return "test-api-key"

    def get_base_url(self) -> str:
        return "http://test-base-url"


@pytest.fixture
def mixin():
    config = RemoteInferenceProviderConfig()
    m = OpenAIMixinTestImpl(config=config)
    m.overwrite_completion_id = False
    return m


class TestIssue3185Regression:
    async def test_streaming_result_has_aclose(self, mixin):
        mock_stream = MockAsyncStream([MockChunk("1")])

        assert not hasattr(mock_stream, "aclose")

        result = await mixin._maybe_overwrite_id(mock_stream, stream=True)

        assert hasattr(result, "aclose"), "Result MUST have aclose() for AsyncIterator"
        assert inspect.isasyncgen(result)
        assert isinstance(result, AsyncIterator)

    async def test_streaming_yields_all_chunks(self, mixin):
        chunks = [MockChunk("1", "a"), MockChunk("2", "b")]
        mock_stream = MockAsyncStream(chunks)

        result = await mixin._maybe_overwrite_id(mock_stream, stream=True)

        received = [c async for c in result]
        assert len(received) == 2
        assert received[0].content == "a"
        assert received[1].content == "b"

    async def test_non_streaming_returns_directly(self, mixin):
        mock_response = MagicMock()
        mock_response.id = "test-id"

        result = await mixin._maybe_overwrite_id(mock_response, stream=False)

        assert result is mock_response
        assert not inspect.isasyncgen(result)


class TestIdOverwriting:
    async def test_ids_overwritten_when_enabled(self):
        config = RemoteInferenceProviderConfig()
        mixin = OpenAIMixinTestImpl(config=config)
        mixin.overwrite_completion_id = True

        chunks = [MockChunk("orig-1"), MockChunk("orig-2")]
        result = await mixin._maybe_overwrite_id(MockAsyncStream(chunks), stream=True)

        received = [c async for c in result]
        assert all(c.id.startswith("cltsd-") for c in received)
        assert received[0].id == received[1].id  # Same ID for all chunks

    async def test_ids_preserved_when_disabled(self):
        config = RemoteInferenceProviderConfig()
        mixin = OpenAIMixinTestImpl(config=config)
        mixin.overwrite_completion_id = False

        chunks = [MockChunk("orig-1"), MockChunk("orig-2")]
        result = await mixin._maybe_overwrite_id(MockAsyncStream(chunks), stream=True)

        received = [c async for c in result]
        assert received[0].id == "orig-1"
        assert received[1].id == "orig-2"
