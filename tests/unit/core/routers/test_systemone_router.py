# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Unit tests for SystemOneRouter: model-based dispatch for POST /v1/systemone (#6746)."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from ogx.core.routers.systemone import SystemOneRouter
from ogx_api import ModelNotFoundError, ModelType, ModelTypeError, RoutingTable, SystemOneRequest, SystemOneResponse


def _request(model: str = "test-decision-model") -> SystemOneRequest:
    return SystemOneRequest(
        model=model,
        state="Help! My payouts have been failing for 3 days.",
        questions={"is_urgent": {"type": "noul", "instructions": "Does this convey urgency?"}},
    )


def _response(model: str = "provider-model-123") -> SystemOneResponse:
    return SystemOneResponse(model=model, answers={"is_urgent": {"type": "noul", "noul": 0.95}})


@pytest.fixture
def mock_routing_table():
    routing_table = MagicMock(spec=RoutingTable)
    routing_table.impls_by_provider_id = {}
    routing_table.policy = []

    mock_model = MagicMock()
    mock_model.identifier = "test-decision-model"
    mock_model.model_type = ModelType.systemone
    mock_model.provider_resource_id = "provider-model-123"

    mock_provider = MagicMock()
    mock_provider.__provider_id__ = "typesafe"
    mock_provider.judge = AsyncMock(return_value=_response())

    routing_table.get_object_by_identifier = AsyncMock(return_value=mock_model)
    routing_table.get_provider_impl = AsyncMock(return_value=mock_provider)

    return routing_table, mock_provider


async def test_judge_resolves_model_and_dispatches_to_provider(mock_routing_table):
    routing_table, provider = mock_routing_table
    router = SystemOneRouter(routing_table)

    result = await router.judge(_request("test-decision-model"))

    routing_table.get_object_by_identifier.assert_called_once_with("model", "test-decision-model")
    provider.judge.assert_called_once()
    sent_request = provider.judge.call_args.args[0]
    assert sent_request.model == "provider-model-123"
    assert result == _response()


async def test_judge_rejects_wrong_model_type(mock_routing_table):
    """Calling /v1/systemone against a non-decision model must raise a clear ModelTypeError."""
    routing_table, _ = mock_routing_table
    routing_table.get_object_by_identifier.return_value.model_type = ModelType.llm
    router = SystemOneRouter(routing_table)

    with pytest.raises(ModelTypeError, match="systemone"):
        await router.judge(_request("test-decision-model"))


async def test_judge_falls_back_to_provider_resource_id_form():
    """model_id in `provider_id/resource_id` form resolves even if not registered as a Model."""
    routing_table = MagicMock(spec=RoutingTable)
    routing_table.policy = []
    mock_provider = MagicMock()
    mock_provider.judge = AsyncMock(return_value=_response())
    routing_table.impls_by_provider_id = {"typesafe": mock_provider}
    routing_table.get_object_by_identifier = AsyncMock(return_value=None)

    router = SystemOneRouter(routing_table)
    result = await router.judge(_request("typesafe/jev-latest"))

    mock_provider.judge.assert_called_once()
    sent_request = mock_provider.judge.call_args.args[0]
    assert sent_request.model == "jev-latest"
    assert result == _response()


async def test_judge_raises_not_found_for_unknown_provider():
    routing_table = MagicMock(spec=RoutingTable)
    routing_table.policy = []
    routing_table.impls_by_provider_id = {}
    routing_table.get_object_by_identifier = AsyncMock(return_value=None)

    router = SystemOneRouter(routing_table)
    with pytest.raises(ModelNotFoundError):
        await router.judge(_request("unknown-model"))
