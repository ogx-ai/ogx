# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Tests for memory.ranking_options on the Responses memory retrieval path."""

from unittest.mock import AsyncMock

import httpx
import pytest

from ogx.core.exceptions import translate_exception
from ogx.providers.inline.responses.builtin.config import MemoryConfig
from ogx.providers.inline.responses.builtin.responses.memory import resolve_memory_context
from ogx.providers.inline.responses.builtin.responses.openai_responses import OpenAIResponsesImpl
from ogx_api import InvalidParameterError
from ogx_api.responses.models import CreateResponseRequest, MemoryToolConfig
from ogx_api.vector_io.models import SearchRankingOptions
from tests.unit.providers.responses.builtin.test_openai_responses_helpers import fake_stream

UNSUPPORTED_HYBRID_SEARCH_MESSAGE = "The provider of vector store 'vs_mem' does not support weighted hybrid search."


def _unsupported_hybrid_search_error() -> InvalidParameterError:
    """The error openai_search_vector_store raises for a store that cannot apply the weights."""
    return InvalidParameterError(
        "ranking_options.hybrid_search",
        {"embedding_weight": 0.3, "text_weight": 0.7},
        UNSUPPORTED_HYBRID_SEARCH_MESSAGE,
    )


def _hybrid_search_memory() -> MemoryToolConfig:
    return MemoryToolConfig(
        owner_id="user-123",
        ranking_options=SearchRankingOptions(hybrid_search={"embedding_weight": 0.3, "text_weight": 0.7}),
    )


async def test_resolve_memory_context_propagates_unsupported_hybrid_search():
    """Test that memory.ranking_options.hybrid_search on an unsupported store fails the request.

    resolve_memory_context absorbs only VectorStoreNotFoundError, so the 400 the vector store raises
    for weights its provider cannot apply reaches responses.create and becomes a Bad Request. That is
    deliberate: answering without the memory the caller asked for would hide the parameter the server
    refused, which is the silent degradation this option must not have.
    """
    vector_io = AsyncMock()
    vector_io.openai_search_vector_store.side_effect = _unsupported_hybrid_search_error()

    with pytest.raises(InvalidParameterError) as excinfo:
        await resolve_memory_context(
            vector_io_api=vector_io,
            memory_config=MemoryConfig(enabled=True, default_vector_store_id="vs_mem"),
            request_memory=_hybrid_search_memory(),
            input="repo prefs",
            metadata=None,
            safety_identifier=None,
        )

    assert translate_exception(excinfo.value).status_code == httpx.codes.BAD_REQUEST
    request = vector_io.openai_search_vector_store.await_args.kwargs["request"]
    assert request.search_mode == "hybrid"
    assert request.ranking_options.hybrid_search.model_dump() == {"embedding_weight": 0.3, "text_weight": 0.7}


async def test_create_response_fails_when_memory_hybrid_search_is_unsupported(mock_responses_store):
    """Test that the 400 above is what a responses.create caller actually gets."""
    vector_io = AsyncMock()
    vector_io.openai_search_vector_store.side_effect = _unsupported_hybrid_search_error()
    inference_api = AsyncMock()
    inference_api.openai_chat_completion.side_effect = [fake_stream()]

    impl = OpenAIResponsesImpl(
        inference_api=inference_api,
        tool_groups_api=AsyncMock(),
        tool_runtime_api=AsyncMock(),
        responses_store=mock_responses_store,
        vector_io_api=vector_io,
        moderation_endpoint=None,
        conversations_api=AsyncMock(),
        prompts_api=AsyncMock(),
        files_api=AsyncMock(),
        connectors_api=AsyncMock(),
        memory_config=MemoryConfig(enabled=True, default_vector_store_id="vs_mem"),
    )

    with pytest.raises(InvalidParameterError) as excinfo:
        await impl.create_openai_response(
            CreateResponseRequest(
                input="repo prefs",
                model="ollama/llama3.2:3b",
                memory=_hybrid_search_memory(),
            )
        )

    http_exc = translate_exception(excinfo.value)
    assert http_exc.status_code == httpx.codes.BAD_REQUEST
    assert "ranking_options.hybrid_search" in http_exc.detail
    assert UNSUPPORTED_HYBRID_SEARCH_MESSAGE in http_exc.detail
    # The rejection happens before the model is asked anything.
    inference_api.openai_chat_completion.assert_not_called()
