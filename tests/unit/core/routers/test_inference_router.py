# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""
Unit tests for InferenceRouter to verify correct provider method invocation.

Test Categories:
1. Rerank method routing - validates that rerank calls are properly routed to providers
2. Model resolution - validates model to provider mapping
3. Parameter transformation - validates request object modifications for provider calls

Specific Tests:
- test_rerank_calls_provider_correctly: Validates the router calls provider.rerank() with correct RerankRequest
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from ogx.core.routers.inference import InferenceRouter
from ogx_api import (
    ModelType,
    OpenAICompletion,
    OpenAICompletionRequestWithExtraBody,
    RerankData,
    RerankResponse,
    RoutingTable,
)
from ogx_api.inference import RerankRequest
from ogx_api.inference.models import OpenAICompletionChoice


@pytest.fixture
def mock_routing_table():
    """Create a mock routing table with model and provider setup"""
    routing_table = MagicMock(spec=RoutingTable)

    mock_model = MagicMock()
    mock_model.identifier = "test-rerank-model"
    mock_model.model_type = ModelType.rerank
    mock_model.provider_resource_id = "provider-rerank-model-123"

    mock_provider = MagicMock()
    mock_provider.__provider_id__ = "test_provider"

    routing_table.get_object_by_identifier = AsyncMock(return_value=mock_model)
    routing_table.get_provider_impl = AsyncMock(return_value=mock_provider)

    return routing_table, mock_provider


@pytest.fixture
def mock_llm_routing_table():
    """Create a mock routing table with an LLM model registered under a fully qualified id"""
    routing_table = MagicMock(spec=RoutingTable)

    mock_model = MagicMock()
    mock_model.identifier = "test_provider/test-llm-model"
    mock_model.model_type = ModelType.llm
    mock_model.provider_resource_id = "test-llm-model"

    mock_provider = MagicMock()
    mock_provider.__provider_id__ = "test_provider"

    routing_table.get_object_by_identifier = AsyncMock(return_value=mock_model)
    routing_table.get_provider_impl = AsyncMock(return_value=mock_provider)

    return routing_table, mock_provider


def _make_completion_chunk(text: str, model: str) -> OpenAICompletion:
    return OpenAICompletion(
        id="cmpl-test",
        choices=[OpenAICompletionChoice(finish_reason="stop", text=text, index=0)],
        created=0,
        model=model,
        object="text_completion",
    )


async def test_openai_completion_streaming_rewrites_model_id(mock_llm_routing_table):
    """
    Test that streamed /v1/completions chunks report the fully qualified model id
    that the client requested, not the provider-internal resource id.

    This mirrors the non-streaming path in openai_completion (which sets
    response.model = request_model_id) and the chat streaming path
    (stream_tokens_and_compute_metrics_openai_chat, which rewrites chunk.model).
    """
    routing_table, mock_provider = mock_llm_routing_table
    router = InferenceRouter(routing_table=routing_table)

    async def provider_stream():
        # Providers respond with their internal model id
        yield _make_completion_chunk("Hello", model="test-llm-model")
        yield _make_completion_chunk(" world", model="test-llm-model")

    mock_provider.openai_completion = AsyncMock(return_value=provider_stream())

    params = OpenAICompletionRequestWithExtraBody(
        model="test_provider/test-llm-model",
        prompt="Say hello",
        stream=True,
    )

    stream = await router.openai_completion(params)
    chunks = [chunk async for chunk in stream]

    assert len(chunks) == 2
    assert [chunk.model for chunk in chunks] == ["test_provider/test-llm-model", "test_provider/test-llm-model"], (
        "Streamed completion chunks should carry the requested model id, not the provider resource id"
    )
    assert [choice.text for chunk in chunks for choice in chunk.choices] == ["Hello", " world"]

    # The provider itself should still be called with its own resource id
    called_params = mock_provider.openai_completion.call_args.args[0]
    assert called_params.model == "test-llm-model"


async def test_openai_completion_streaming_empty_stream(mock_llm_routing_table):
    """A provider stream that yields no chunks produces an empty stream without errors."""
    routing_table, mock_provider = mock_llm_routing_table
    router = InferenceRouter(routing_table=routing_table)

    async def provider_stream():
        return
        yield  # unreachable; makes this function an async generator

    mock_provider.openai_completion = AsyncMock(return_value=provider_stream())

    params = OpenAICompletionRequestWithExtraBody(
        model="test_provider/test-llm-model",
        prompt="Say hello",
        stream=True,
    )

    stream = await router.openai_completion(params)
    chunks = [chunk async for chunk in stream]

    assert chunks == []


async def test_openai_completion_streaming_model_id_already_correct(mock_llm_routing_table):
    """Chunks that already carry the fully qualified model id are passed through unchanged."""
    routing_table, mock_provider = mock_llm_routing_table
    router = InferenceRouter(routing_table=routing_table)

    async def provider_stream():
        yield _make_completion_chunk("Hello", model="test_provider/test-llm-model")

    mock_provider.openai_completion = AsyncMock(return_value=provider_stream())

    params = OpenAICompletionRequestWithExtraBody(
        model="test_provider/test-llm-model",
        prompt="Say hello",
        stream=True,
    )

    stream = await router.openai_completion(params)
    chunks = [chunk async for chunk in stream]

    assert len(chunks) == 1
    assert chunks[0].model == "test_provider/test-llm-model"
    assert chunks[0].choices[0].text == "Hello"


async def test_openai_completion_streaming_skips_none_chunks(mock_llm_routing_table):
    """None chunks from a provider are skipped, mirroring the chat streaming path."""
    routing_table, mock_provider = mock_llm_routing_table
    router = InferenceRouter(routing_table=routing_table)

    async def provider_stream():
        yield _make_completion_chunk("Hello", model="test-llm-model")
        yield None
        yield _make_completion_chunk(" world", model="test-llm-model")

    mock_provider.openai_completion = AsyncMock(return_value=provider_stream())

    params = OpenAICompletionRequestWithExtraBody(
        model="test_provider/test-llm-model",
        prompt="Say hello",
        stream=True,
    )

    stream = await router.openai_completion(params)
    chunks = [chunk async for chunk in stream]

    assert [chunk.model for chunk in chunks] == ["test_provider/test-llm-model", "test_provider/test-llm-model"]
    assert [choice.text for chunk in chunks for choice in chunk.choices] == ["Hello", " world"]


async def test_openai_completion_streaming_propagates_provider_errors(mock_llm_routing_table):
    """Errors raised by the provider mid-stream propagate to the caller after earlier chunks are delivered."""
    routing_table, mock_provider = mock_llm_routing_table
    router = InferenceRouter(routing_table=routing_table)

    async def provider_stream():
        yield _make_completion_chunk("Hello", model="test-llm-model")
        raise RuntimeError("provider stream failed")

    mock_provider.openai_completion = AsyncMock(return_value=provider_stream())

    params = OpenAICompletionRequestWithExtraBody(
        model="test_provider/test-llm-model",
        prompt="Say hello",
        stream=True,
    )

    stream = await router.openai_completion(params)
    chunks = []
    with pytest.raises(RuntimeError, match="provider stream failed"):
        async for chunk in stream:
            chunks.append(chunk)

    assert len(chunks) == 1
    assert chunks[0].model == "test_provider/test-llm-model"


async def test_openai_completion_non_streaming_rewrites_model_id(mock_llm_routing_table):
    """Non-streaming /v1/completions responses report the requested model id (regression guard)."""
    routing_table, mock_provider = mock_llm_routing_table
    router = InferenceRouter(routing_table=routing_table)

    mock_provider.openai_completion = AsyncMock(
        return_value=_make_completion_chunk("Hello world", model="test-llm-model")
    )

    params = OpenAICompletionRequestWithExtraBody(
        model="test_provider/test-llm-model",
        prompt="Say hello",
    )

    response = await router.openai_completion(params)

    assert response.model == "test_provider/test-llm-model"


async def test_rerank_calls_provider_correctly(mock_routing_table):
    """
    Test that InferenceRouter.rerank() calls the provider's rerank method with the correct RerankRequest.

    This test validates:
    - The provider's rerank method is called exactly once
    - The provider receives a RerankRequest object (not individual parameters)
    - The model ID is substituted with provider_resource_id
    """
    routing_table, mock_provider = mock_routing_table
    router = InferenceRouter(routing_table=routing_table)

    expected_response = RerankResponse(
        data=[
            RerankData(index=0, relevance_score=0.9),
        ]
    )
    mock_provider.rerank = AsyncMock(return_value=expected_response)

    request = RerankRequest(
        model="test-rerank-model",
        query="test query",
        items=["item1", "item2"],
        max_num_results=1,
    )

    result = await router.rerank(request)

    mock_provider.rerank.assert_called_once()

    call_args = mock_provider.rerank.call_args
    assert len(call_args.args) == 1, "Provider.rerank should be called with exactly one argument"
    assert isinstance(call_args.args[0], RerankRequest), "Provider.rerank should receive a RerankRequest object"

    called_request = call_args.args[0]
    assert called_request.model == "provider-rerank-model-123", "Model should be substituted with provider_resource_id"

    assert called_request.query == "test query"
    assert called_request.items == ["item1", "item2"]
    assert called_request.max_num_results == 1

    assert result == expected_response
