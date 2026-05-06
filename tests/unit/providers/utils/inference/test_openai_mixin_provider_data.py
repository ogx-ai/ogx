# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import json
from collections.abc import Iterable
from unittest.mock import MagicMock, Mock

import pytest

from ogx.core.request_headers import request_provider_data_context
from ogx.providers.utils.inference.model_registry import RemoteInferenceProviderConfig
from tests.unit.providers.utils.inference.openai_mixin_helpers import (
    CustomListProviderModelIdsImplementation,
    OpenAIMixinImpl,
    OpenAIMixinWithProviderData,
    ProviderDataValidator,  # noqa: F401 — referenced by provider_data_validator string path
)


class TestOpenAIMixinCustomListProviderModelIds:
    """Test cases for custom list_provider_model_ids() implementation functionality"""

    @pytest.fixture
    def custom_model_ids_list(self):
        """Create a list of custom model ID strings"""
        return ["custom-model-1", "custom-model-2", "custom-embedding"]

    @pytest.fixture
    def config(self):
        """Create RemoteInferenceProviderConfig instance"""
        return RemoteInferenceProviderConfig()

    @pytest.fixture
    def adapter(self, custom_model_ids_list, config):
        """Create mixin instance with custom list_provider_model_ids implementation"""
        mixin = CustomListProviderModelIdsImplementation(config=config, custom_model_ids=custom_model_ids_list)
        mixin.embedding_model_metadata = {"custom-embedding": {"embedding_dimension": 768, "context_length": 512}}
        return mixin

    async def test_is_used(self, adapter, custom_model_ids_list):
        """Test that custom list_provider_model_ids() implementation is used instead of client.models.list()"""
        result = await adapter.list_models()

        assert result is not None
        assert len(result) == 3

        assert set(custom_model_ids_list) == {m.identifier for m in result}

    async def test_populates_cache(self, adapter, custom_model_ids_list):
        """Test that custom list_provider_model_ids() results are cached"""
        assert len(adapter._model_cache) == 0

        await adapter.list_models()

        assert set(custom_model_ids_list) == set(adapter._model_cache.keys())

    async def test_respects_allowed_models(self, config):
        """Test that custom list_provider_model_ids() respects allowed_models filtering"""
        mixin = CustomListProviderModelIdsImplementation(
            config=config, custom_model_ids=["model-1", "model-2", "model-3"]
        )
        mixin.config.allowed_models = ["model-1"]

        result = await mixin.list_models()

        assert result is not None
        assert len(result) == 1
        assert result[0].identifier == "model-1"

    async def test_with_empty_list(self, config):
        """Test that custom list_provider_model_ids() handles empty list correctly"""
        mixin = CustomListProviderModelIdsImplementation(config=config, custom_model_ids=[])

        result = await mixin.list_models()

        assert result is not None
        assert len(result) == 0
        assert len(mixin._model_cache) == 0

    async def test_wrong_type_raises_error(self, config):
        """Test that list_provider_model_ids() returning unhashable items results in an error"""
        mixin = CustomListProviderModelIdsImplementation(
            config=config, custom_model_ids=["valid-model", ["nested", "list"]]
        )
        with pytest.raises(Exception, match="is not a string"):
            await mixin.list_models()

        mixin = CustomListProviderModelIdsImplementation(
            config=config, custom_model_ids=[{"key": "value"}, "valid-model"]
        )
        with pytest.raises(Exception, match="is not a string"):
            await mixin.list_models()

        mixin = CustomListProviderModelIdsImplementation(config=config, custom_model_ids=["valid-model", 42.0])
        with pytest.raises(Exception, match="is not a string"):
            await mixin.list_models()

        mixin = CustomListProviderModelIdsImplementation(config=config, custom_model_ids=[None])
        with pytest.raises(Exception, match="is not a string"):
            await mixin.list_models()

    async def test_non_iterable_raises_error(self, config):
        """Test that list_provider_model_ids() returning non-iterable type raises error"""
        mixin = CustomListProviderModelIdsImplementation(config=config, custom_model_ids=42)

        with pytest.raises(
            TypeError,
            match=r"Failed to list models: CustomListProviderModelIdsImplementation\.list_provider_model_ids\(\) must return an iterable.*but returned int",
        ):
            await mixin.list_models()

    async def test_accepts_various_iterables(self, config):
        """Test that list_provider_model_ids() accepts tuples, sets, generators, etc."""

        tuples = CustomListProviderModelIdsImplementation(
            config=config, custom_model_ids=("model-1", "model-2", "model-3")
        )
        result = await tuples.list_models()
        assert result is not None
        assert len(result) == 3

        class GeneratorAdapter(OpenAIMixinImpl):
            async def list_provider_model_ids(self) -> Iterable[str]:
                def gen():
                    yield "gen-model-1"
                    yield "gen-model-2"

                return gen()

        mixin = GeneratorAdapter(config=config)
        result = await mixin.list_models()
        assert result is not None
        assert len(result) == 2

        sets = CustomListProviderModelIdsImplementation(config=config, custom_model_ids={"set-model-1", "set-model-2"})
        result = await sets.list_models()
        assert result is not None
        assert len(result) == 2


class TestOpenAIMixinProviderDataApiKey:
    """Test cases for provider_data_api_key_field functionality"""

    @pytest.fixture
    def mixin_with_provider_data_field(self):
        """Mixin instance with provider_data_api_key_field set"""
        config = RemoteInferenceProviderConfig()
        mixin_instance = OpenAIMixinWithProviderData(config=config)

        # Mock provider_spec for provider data validation
        mock_provider_spec = MagicMock()
        mock_provider_spec.provider_type = "test-provider-with-data"
        mock_provider_spec.provider_data_validator = (
            "tests.unit.providers.utils.inference.openai_mixin_helpers.ProviderDataValidator"
        )
        mixin_instance.__provider_spec__ = mock_provider_spec

        return mixin_instance

    @pytest.fixture
    def mixin_with_provider_data_field_and_none_api_key(self, mixin_with_provider_data_field):
        mixin_with_provider_data_field.get_api_key = Mock(return_value=None)
        return mixin_with_provider_data_field

    def test_no_provider_data(self, mixin_with_provider_data_field):
        """Test that client uses config API key when no provider data is available"""
        assert mixin_with_provider_data_field.client.api_key == "default-api-key"

    def test_with_provider_data(self, mixin_with_provider_data_field):
        """Test that provider data API key overrides config API key"""
        with request_provider_data_context({"x-ogx-provider-data": json.dumps({"test_api_key": "provider-data-key"})}):
            assert mixin_with_provider_data_field.client.api_key == "provider-data-key"

    def test_with_wrong_key(self, mixin_with_provider_data_field):
        """Test fallback to config when provider data doesn't have the required key"""
        with request_provider_data_context({"x-ogx-provider-data": json.dumps({"wrong_key": "some-value"})}):
            assert mixin_with_provider_data_field.client.api_key == "default-api-key"

    def test_error_when_no_config_and_provider_data_has_wrong_key(
        self, mixin_with_provider_data_field_and_none_api_key
    ):
        """Test that ValueError is raised when provider data exists but doesn't have required key"""
        with request_provider_data_context({"x-ogx-provider-data": json.dumps({"wrong_key": "some-value"})}):
            with pytest.raises(ValueError, match="API key not provided"):
                _ = mixin_with_provider_data_field_and_none_api_key.client

    def test_error_message_includes_correct_field_names(self, mixin_with_provider_data_field_and_none_api_key):
        """Test that error message includes correct field name and header information"""
        with pytest.raises(ValueError) as exc_info:
            _ = mixin_with_provider_data_field_and_none_api_key.client

        error_message = str(exc_info.value)
        assert "test_api_key" in error_message
        assert "x-ogx-provider-data" in error_message
