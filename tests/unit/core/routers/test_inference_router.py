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
