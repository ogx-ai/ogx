# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Unit tests for provider_model_id="auto" model resolution in ModelsRoutingTable."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from ogx.core.routing_tables.models import ModelsRoutingTable
from ogx_api import Model, ModelType


@pytest.fixture
def mock_provider():
    """Create a mock provider with list_models capability."""
    provider = AsyncMock()
    return provider


@pytest.fixture
def routing_table(mock_provider):
    """Create a ModelsRoutingTable with a mock provider."""
    table = ModelsRoutingTable(
        impls_by_provider_id={"test-provider": mock_provider},
        policy=MagicMock(),
        dist_registry=MagicMock(),
    )
    return table


class TestAutoModelResolution:
    async def test_resolves_to_first_matching_model(self, routing_table, mock_provider):
        """Resolves provider_model_id="auto" to the first model matching the type."""
        # Setup mock provider to return multiple models
        mock_models = [
            Model(
                identifier="test-provider/model-1",
                provider_resource_id="model-1",
                provider_id="test-provider",
                model_type=ModelType.llm,
            ),
            Model(
                identifier="test-provider/model-2",
                provider_resource_id="model-2",
                provider_id="test-provider",
                model_type=ModelType.llm,
            ),
        ]
        mock_provider.list_models.return_value = mock_models

        result = await routing_table._resolve_auto_model("test-provider", ModelType.llm)

        assert result == "model-1"
        mock_provider.list_models.assert_called_once()

    async def test_filters_by_model_type(self, routing_table, mock_provider):
        """Only returns models matching the requested model_type."""
        mock_models = [
            Model(
                identifier="test-provider/embedding-model",
                provider_resource_id="embedding-model",
                provider_id="test-provider",
                model_type=ModelType.embedding,
            ),
            Model(
                identifier="test-provider/llm-model",
                provider_resource_id="llm-model",
                provider_id="test-provider",
                model_type=ModelType.llm,
            ),
        ]
        mock_provider.list_models.return_value = mock_models

        result = await routing_table._resolve_auto_model("test-provider", ModelType.llm)

        assert result == "llm-model"

    async def test_raises_on_provider_not_found(self, routing_table):
        """Raises ValueError if provider doesn't exist."""
        with pytest.raises(ValueError, match="Provider 'unknown' not found"):
            await routing_table._resolve_auto_model("unknown", ModelType.llm)

    async def test_raises_on_list_models_failure(self, routing_table, mock_provider):
        """Raises ValueError if provider.list_models() fails."""
        mock_provider.list_models.side_effect = Exception("Connection error")

        with pytest.raises(ValueError, match="Failed to list models"):
            await routing_table._resolve_auto_model("test-provider", ModelType.llm)

    async def test_raises_on_no_models(self, routing_table, mock_provider):
        """Raises ValueError if provider returns empty model list."""
        mock_provider.list_models.return_value = []

        with pytest.raises(ValueError, match="returned no models"):
            await routing_table._resolve_auto_model("test-provider", ModelType.llm)

    async def test_raises_on_no_matching_type(self, routing_table, mock_provider):
        """Raises ValueError if no models match the requested type."""
        mock_models = [
            Model(
                identifier="test-provider/embedding-model",
                provider_resource_id="embedding-model",
                provider_id="test-provider",
                model_type=ModelType.embedding,
            ),
        ]
        mock_provider.list_models.return_value = mock_models

        with pytest.raises(ValueError, match="No llm models found"):
            await routing_table._resolve_auto_model("test-provider", ModelType.llm)


class TestRegisterModelWithAuto:
    async def test_register_model_resolves_auto(self, routing_table, mock_provider):
        """register_model resolves provider_model_id="auto" to an actual model."""
        mock_models = [
            Model(
                identifier="test-provider/actual-model",
                provider_resource_id="actual-model",
                provider_id="test-provider",
                model_type=ModelType.llm,
            ),
        ]
        mock_provider.list_models.return_value = mock_models

        # Mock the register_object method to avoid database operations
        routing_table.register_object = AsyncMock(
            return_value=Model(
                identifier="test-provider/actual-model",
                provider_resource_id="actual-model",
                provider_id="test-provider",
                model_type=ModelType.llm,
            )
        )

        await routing_table.register_model(
            model_id="claude-haiku",
            provider_model_id="auto",
            provider_id="test-provider",
            model_type=ModelType.llm,
        )

        # Verify the resolved model was registered
        call_args = routing_table.register_object.call_args[0][0]
        assert call_args.provider_resource_id == "actual-model"
        assert call_args.identifier == "test-provider/claude-haiku"  # Uses model_id, not provider_model_id
