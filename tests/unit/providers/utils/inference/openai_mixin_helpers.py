# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from collections.abc import Iterable
from typing import Any
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import pytest
from pydantic import BaseModel, Field, SecretStr

from ogx.providers.utils.inference.model_registry import RemoteInferenceProviderConfig
from ogx.providers.utils.inference.openai_mixin import OpenAIMixin
from ogx_api import Model, ModelType


class OpenAIMixinImpl(OpenAIMixin):
    __provider_id__: str = "test-provider"

    def get_api_key(self) -> str:
        return "test-api-key"

    def get_base_url(self) -> str:
        return "http://test-base-url"


class OpenAIMixinWithEmbeddingsImpl(OpenAIMixinImpl):
    """Test implementation with embedding model metadata"""

    embedding_model_metadata: dict[str, dict[str, int]] = {
        "text-embedding-3-small": {"embedding_dimension": 1536, "context_length": 8192},
        "text-embedding-ada-002": {"embedding_dimension": 1536, "context_length": 8192},
    }


class OpenAIMixinWithCustomModelConstruction(OpenAIMixinImpl):
    """Test implementation that uses construct_model_from_identifier to add rerank models"""

    embedding_model_metadata: dict[str, dict[str, int]] = {
        "text-embedding-3-small": {"embedding_dimension": 1536, "context_length": 8192},
        "text-embedding-ada-002": {"embedding_dimension": 1536, "context_length": 8192},
    }

    # Adds rerank models via construct_model_from_identifier
    rerank_model_ids: set[str] = {"rerank-model-1", "rerank-model-2"}

    def construct_model_from_identifier(self, identifier: str) -> Model:
        if identifier in self.rerank_model_ids:
            return Model(
                provider_id=self.__provider_id__,  # type: ignore[attr-defined]
                provider_resource_id=identifier,
                identifier=identifier,
                model_type=ModelType.rerank,
            )
        return super().construct_model_from_identifier(identifier)


class ProviderDataValidator(BaseModel):
    """Validator for provider data in tests"""

    test_api_key: SecretStr | None = Field(default=None)


class OpenAIMixinWithProviderData(OpenAIMixinImpl):
    """Test implementation that supports provider data API key field"""

    provider_data_api_key_field: str = "test_api_key"

    def get_api_key(self) -> str:
        return "default-api-key"

    def get_base_url(self):
        return "default-base-url"


class CustomListProviderModelIdsImplementation(OpenAIMixinImpl):
    """Test implementation with custom list_provider_model_ids override"""

    custom_model_ids: Any

    async def list_provider_model_ids(self) -> Iterable[str]:
        """Return custom model IDs list"""
        return self.custom_model_ids


@pytest.fixture
def mixin():
    """Create a test instance of OpenAIMixin with mocked model_store"""
    config = RemoteInferenceProviderConfig()
    mixin_instance = OpenAIMixinImpl(config=config)

    # Mock model_store with async methods
    mock_model_store = AsyncMock()
    mock_model = MagicMock()
    mock_model.provider_resource_id = "test-provider-resource-id"
    mock_model_store.get_model = AsyncMock(return_value=mock_model)
    mock_model_store.has_model = AsyncMock(return_value=False)  # Default to False, tests can override
    mixin_instance.model_store = mock_model_store

    return mixin_instance


@pytest.fixture
def mixin_with_embeddings():
    """Create a test instance of OpenAIMixin with embedding model metadata"""
    config = RemoteInferenceProviderConfig()
    return OpenAIMixinWithEmbeddingsImpl(config=config)


@pytest.fixture
def mixin_with_custom_model_construction():
    """Create a test instance using custom construct_model_from_identifier"""
    config = RemoteInferenceProviderConfig()
    return OpenAIMixinWithCustomModelConstruction(config=config)


@pytest.fixture
def mock_models():
    """Create multiple mock OpenAI model objects"""
    models = [MagicMock(id=id) for id in ["some-mock-model-id", "another-mock-model-id", "final-mock-model-id"]]
    return models


@pytest.fixture
def mock_client_with_models(mock_models):
    """Create a mock client with models.list() set up to return mock_models"""
    mock_client = MagicMock()

    async def mock_models_list():
        for model in mock_models:
            yield model

    mock_client.models.list.return_value = mock_models_list()
    return mock_client


@pytest.fixture
def mock_client_with_empty_models():
    """Create a mock client with models.list() set up to return empty list"""
    mock_client = MagicMock()

    async def mock_empty_models_list():
        return
        yield  # Make it an async generator but don't yield anything

    mock_client.models.list.return_value = mock_empty_models_list()
    return mock_client


@pytest.fixture
def mock_client_with_exception():
    """Create a mock client with models.list() set up to raise an exception"""
    mock_client = MagicMock()
    mock_client.models.list.side_effect = Exception("API Error")
    return mock_client


@pytest.fixture
def mock_client_context():
    """Fixture that provides a context manager for mocking the OpenAI client"""

    def _mock_client_context(mixin, mock_client):
        return patch.object(type(mixin), "client", new_callable=PropertyMock, return_value=mock_client)

    return _mock_client_context


def _assert_models_match_expected(actual_models, expected_models):
    """Verify the models match expected attributes.

    Args:
        actual_models: List of models to verify
        expected_models: Mapping of model identifier to expected attribute values
    """
    for identifier, expected_attrs in expected_models.items():
        model = next(m for m in actual_models if m.identifier == identifier)
        for attr_name, expected_value in expected_attrs.items():
            assert getattr(model, attr_name) == expected_value
