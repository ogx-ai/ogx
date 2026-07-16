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
4. Compression - validates the InferenceRouter compression hook (no-op when disabled,
   windowing/dropping oldest turns when over budget, summarizing dropped turns via a
   direct provider call that bypasses the router itself)

Specific Tests:
- test_rerank_calls_provider_correctly: Validates the router calls provider.rerank() with correct RerankRequest
"""

import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from ogx.core.datatypes import CompressionConfig
from ogx.core.routers.inference import InferenceRouter
from ogx_api import (
    ModelType,
    OpenAIAssistantMessageParam,
    OpenAIChatCompletion,
    OpenAIChatCompletionRequestWithExtraBody,
    OpenAIChatCompletionResponseMessage,
    OpenAIChoice,
    OpenAIUserMessageParam,
    RerankData,
    RerankResponse,
    RoutingTable,
)
from ogx_api.inference import RerankRequest


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


@pytest.fixture
def mock_chat_routing_table():
    """Create a mock routing table with an LLM model and provider setup for chat completion tests."""
    routing_table = MagicMock(spec=RoutingTable)

    mock_model = MagicMock()
    mock_model.identifier = "test-chat-model"
    mock_model.model_type = ModelType.llm
    mock_model.provider_resource_id = "provider-chat-model-123"

    mock_provider = MagicMock()
    mock_provider.__provider_id__ = "test_provider"

    routing_table.get_object_by_identifier = AsyncMock(return_value=mock_model)
    routing_table.get_provider_impl = AsyncMock(return_value=mock_provider)

    return routing_table, mock_provider


def _chat_completion_response(content: str = "ok") -> OpenAIChatCompletion:
    return OpenAIChatCompletion(
        id="resp_1",
        model="provider-chat-model-123",
        created=int(time.time()),
        choices=[
            OpenAIChoice(index=0, finish_reason="stop", message=OpenAIChatCompletionResponseMessage(content=content))
        ],
    )


async def test_compression_disabled_leaves_messages_unchanged(mock_chat_routing_table):
    """No CompressionConfig configured: the provider should receive the request untouched."""
    routing_table, mock_provider = mock_chat_routing_table
    router = InferenceRouter(routing_table=routing_table, compression_config=None)
    mock_provider.openai_chat_completion = AsyncMock(return_value=_chat_completion_response())

    request = OpenAIChatCompletionRequestWithExtraBody(
        model="test-chat-model",
        messages=[OpenAIUserMessageParam(content="hello")],
        stream=False,
    )

    await router.openai_chat_completion(request)

    called_request = mock_provider.openai_chat_completion.call_args.args[0]
    assert len(called_request.messages) == 1
    assert called_request.messages[0].content == "hello"


async def test_compression_windows_out_oldest_turns(mock_chat_routing_table):
    """With a small max_context_tokens budget, the oldest turn should be dropped before
    the request reaches the provider."""
    routing_table, mock_provider = mock_chat_routing_table
    compression_config = CompressionConfig(enabled=True, max_context_tokens=20, dedupe_tool_outputs=False)
    router = InferenceRouter(routing_table=routing_table, compression_config=compression_config)
    mock_provider.openai_chat_completion = AsyncMock(return_value=_chat_completion_response())

    request = OpenAIChatCompletionRequestWithExtraBody(
        model="test-chat-model",
        messages=[
            OpenAIUserMessageParam(content="turn one " * 20),
            OpenAIAssistantMessageParam(content="reply one " * 20),
            OpenAIUserMessageParam(content="turn two"),
        ],
        stream=False,
    )

    await router.openai_chat_completion(request)

    called_request = mock_provider.openai_chat_completion.call_args.args[0]
    assert len(called_request.messages) == 1
    assert called_request.messages[0].content == "turn two"


async def test_compression_summarizes_dropped_turns_via_direct_provider_call(mock_chat_routing_table):
    """With summarize_dropped_turns enabled, the router should call the provider twice:
    once internally to summarize the dropped turns (bypassing the router itself so that
    call isn't recursively compressed), and once for the real request with the summary
    spliced in."""
    routing_table, mock_provider = mock_chat_routing_table
    compression_config = CompressionConfig(enabled=True, max_context_tokens=20, summarize_dropped_turns=True)
    router = InferenceRouter(routing_table=routing_table, compression_config=compression_config)

    call_count = 0

    async def fake_openai_chat_completion(params):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return _chat_completion_response(content="short summary")
        return _chat_completion_response(content="final answer")

    mock_provider.openai_chat_completion = AsyncMock(side_effect=fake_openai_chat_completion)

    request = OpenAIChatCompletionRequestWithExtraBody(
        model="test-chat-model",
        messages=[
            OpenAIUserMessageParam(content="turn one " * 20),
            OpenAIAssistantMessageParam(content="reply one " * 20),
            OpenAIUserMessageParam(content="turn two"),
        ],
        stream=False,
    )

    await router.openai_chat_completion(request)

    assert mock_provider.openai_chat_completion.call_count == 2

    final_request = mock_provider.openai_chat_completion.call_args_list[1].args[0]
    assert final_request.messages[0].role == "system"
    assert "short summary" in final_request.messages[0].content
    assert final_request.messages[1].content == "turn two"
