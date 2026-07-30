# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import pytest

from ogx.providers.utils.inference.model_registry import RemoteInferenceProviderConfig
from ogx_api import (
    Model,
    ModelType,
    OpenAIChatCompletionRequestWithExtraBody,
    OpenAIUserMessageParam,
)
from tests.unit.providers.utils.inference.openai_mixin_helpers import (
    OpenAIMixinImpl,
    _assert_models_match_expected,
)


class TestOpenAIMixinListModels:
    """Test cases for the list_models method"""

    async def test_list_models_success(self, mixin, mock_client_with_models, mock_client_context):
        """Test successful model listing"""
        assert len(mixin._model_cache) == 0

        with mock_client_context(mixin, mock_client_with_models):
            result = await mixin.list_models()

            assert result is not None
            assert len(result) == 3

            model_ids = [model.identifier for model in result]
            assert "some-mock-model-id" in model_ids
            assert "another-mock-model-id" in model_ids
            assert "final-mock-model-id" in model_ids

            for model in result:
                assert model.provider_id == "test-provider"
                assert model.model_type == ModelType.llm
                assert model.provider_resource_id == model.identifier

            assert len(mixin._model_cache) == 3
            for model_id in ["some-mock-model-id", "another-mock-model-id", "final-mock-model-id"]:
                assert model_id in mixin._model_cache
                cached_model = mixin._model_cache[model_id]
                assert cached_model.identifier == model_id
                assert cached_model.provider_resource_id == model_id

    async def test_list_models_empty_response(self, mixin, mock_client_with_empty_models, mock_client_context):
        """Test handling of empty model list"""
        with mock_client_context(mixin, mock_client_with_empty_models):
            result = await mixin.list_models()

            assert result is not None
            assert len(result) == 0
            assert len(mixin._model_cache) == 0


class TestOpenAIMixinCheckModelAvailability:
    """Test cases for the check_model_availability method"""

    async def test_check_model_availability_with_cache(self, mixin, mock_client_with_models, mock_client_context):
        """Test model availability check when cache is populated"""
        with mock_client_context(mixin, mock_client_with_models):
            mock_client_with_models.models.list.assert_not_called()
            await mixin.list_models()
            mock_client_with_models.models.list.assert_called_once()

            assert await mixin.check_model_availability("some-mock-model-id")
            assert await mixin.check_model_availability("another-mock-model-id")
            assert await mixin.check_model_availability("final-mock-model-id")
            assert not await mixin.check_model_availability("non-existent-model")
            mock_client_with_models.models.list.assert_called_once()

    async def test_check_model_availability_without_cache(self, mixin, mock_client_with_models, mock_client_context):
        """Test model availability check when cache is empty (calls list_models)"""
        assert len(mixin._model_cache) == 0

        with mock_client_context(mixin, mock_client_with_models):
            mock_client_with_models.models.list.assert_not_called()
            assert await mixin.check_model_availability("some-mock-model-id")
            mock_client_with_models.models.list.assert_called_once()

            assert len(mixin._model_cache) == 3
            assert "some-mock-model-id" in mixin._model_cache

    async def test_check_model_availability_model_not_found(self, mixin, mock_client_with_models, mock_client_context):
        """Test model availability check for non-existent model"""
        with mock_client_context(mixin, mock_client_with_models):
            mock_client_with_models.models.list.assert_not_called()
            assert not await mixin.check_model_availability("non-existent-model")
            mock_client_with_models.models.list.assert_called_once()

            assert len(mixin._model_cache) == 3

    async def test_check_model_availability_with_pre_registered_model(
        self, mixin, mock_client_with_models, mock_client_context
    ):
        """Test that check_model_availability returns True for pre-registered models in model_store"""
        # Mock model_store.has_model to return True for a specific model
        mock_model_store = AsyncMock()
        mock_model_store.has_model = AsyncMock(return_value=True)
        mixin.model_store = mock_model_store

        # Test that pre-registered model is found without calling the provider's API
        with mock_client_context(mixin, mock_client_with_models):
            mock_client_with_models.models.list.assert_not_called()
            assert await mixin.check_model_availability("pre-registered-model")
            # Should not call the provider's list_models since model was found in store
            mock_client_with_models.models.list.assert_not_called()
            mock_model_store.has_model.assert_called_once_with("test-provider/pre-registered-model")

    async def test_check_model_availability_fallback_to_provider_when_not_in_store(
        self, mixin, mock_client_with_models, mock_client_context
    ):
        """Test that check_model_availability falls back to provider when model not in store"""
        # Mock model_store.has_model to return False
        mock_model_store = AsyncMock()
        mock_model_store.has_model = AsyncMock(return_value=False)
        mixin.model_store = mock_model_store

        # Test that it falls back to provider's model cache
        with mock_client_context(mixin, mock_client_with_models):
            mock_client_with_models.models.list.assert_not_called()
            assert await mixin.check_model_availability("some-mock-model-id")
            # Should call the provider's list_models since model was not found in store
            mock_client_with_models.models.list.assert_called_once()
            mock_model_store.has_model.assert_called_once_with("test-provider/some-mock-model-id")


class TestOpenAIMixinCacheBehavior:
    """Test cases for cache behavior and edge cases"""

    async def test_cache_overwrites_on_list_models_call(self, mixin, mock_client_with_models, mock_client_context):
        """Test that calling list_models overwrites existing cache"""
        initial_model = Model(
            provider_id="test-provider",
            provider_resource_id="old-model",
            identifier="old-model",
            model_type=ModelType.llm,
        )
        mixin._model_cache = {"old-model": initial_model}

        with mock_client_context(mixin, mock_client_with_models):
            await mixin.list_models()

            assert len(mixin._model_cache) == 3
            assert "old-model" not in mixin._model_cache
            assert "some-mock-model-id" in mixin._model_cache
            assert "another-mock-model-id" in mixin._model_cache
            assert "final-mock-model-id" in mixin._model_cache


class TestOpenAIMixinImagePreprocessing:
    """Test cases for image preprocessing functionality"""

    async def test_openai_chat_completion_with_image_preprocessing_enabled(self, mixin):
        """Test that image URLs are converted to base64 when download_images is True"""
        mixin.download_images = True

        message = OpenAIUserMessageParam(
            role="user",
            content=[
                {"type": "text", "text": "What's in this image?"},
                {"type": "image_url", "image_url": {"url": "http://example.com/image.jpg"}},
            ],
        )

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        with patch.object(type(mixin), "client", new_callable=PropertyMock, return_value=mock_client):
            with patch("ogx.providers.utils.inference.openai_mixin.localize_image_content") as mock_localize:
                mock_localize.return_value = (b"fake_image_data", "jpeg")

                params = OpenAIChatCompletionRequestWithExtraBody(model="test-model", messages=[message])
                await mixin.openai_chat_completion(params)

            mock_localize.assert_called_once_with("http://example.com/image.jpg")

            mock_client.chat.completions.create.assert_called_once()
            call_args = mock_client.chat.completions.create.call_args
            processed_messages = call_args[1]["messages"]
            assert len(processed_messages) == 1
            content = processed_messages[0]["content"]
            assert len(content) == 2
            assert content[0]["type"] == "text"
            assert content[1]["type"] == "image_url"
            assert content[1]["image_url"]["url"] == "data:image/jpeg;base64,ZmFrZV9pbWFnZV9kYXRh"

    async def test_openai_chat_completion_with_image_preprocessing_disabled(self, mixin):
        """Test that image URLs are not modified when download_images is False"""
        mixin.download_images = False  # explicitly set to False

        message = OpenAIUserMessageParam(
            role="user",
            content=[
                {"type": "text", "text": "What's in this image?"},
                {"type": "image_url", "image_url": {"url": "http://example.com/image.jpg"}},
            ],
        )

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        with patch.object(type(mixin), "client", new_callable=PropertyMock, return_value=mock_client):
            with patch("ogx.providers.utils.inference.openai_mixin.localize_image_content") as mock_localize:
                params = OpenAIChatCompletionRequestWithExtraBody(model="test-model", messages=[message])
                await mixin.openai_chat_completion(params)

            mock_localize.assert_not_called()

            mock_client.chat.completions.create.assert_called_once()
            call_args = mock_client.chat.completions.create.call_args
            processed_messages = call_args[1]["messages"]
            assert len(processed_messages) == 1
            content = processed_messages[0]["content"]
            assert len(content) == 2
            assert content[1]["image_url"]["url"] == "http://example.com/image.jpg"


class TestOpenAIMixinEmbeddingModelMetadata:
    """Test cases for embedding_model_metadata attribute functionality"""

    async def test_embedding_model_identified_and_augmented(self, mixin_with_embeddings, mock_client_context):
        """Test that models in embedding_model_metadata are correctly identified as embeddings with metadata"""
        # Create mock models: 1 embedding model and 1 LLM, while there are 2 known embedding models
        mock_embedding_model = MagicMock(id="text-embedding-3-small")
        mock_llm_model = MagicMock(id="gpt-4")
        mock_models = [mock_embedding_model, mock_llm_model]

        mock_client = MagicMock()

        async def mock_models_list():
            for model in mock_models:
                yield model

        mock_client.models.list.return_value = mock_models_list()

        with mock_client_context(mixin_with_embeddings, mock_client):
            result = await mixin_with_embeddings.list_models()

            assert result is not None
            assert len(result) == 2

            expected_models = {
                "text-embedding-3-small": {
                    "model_type": ModelType.embedding,
                    "metadata": {"embedding_dimension": 1536, "context_length": 8192},
                    "provider_id": "test-provider",
                    "provider_resource_id": "text-embedding-3-small",
                },
                "gpt-4": {
                    "model_type": ModelType.llm,
                    "metadata": {},
                    "provider_id": "test-provider",
                    "provider_resource_id": "gpt-4",
                },
            }

            _assert_models_match_expected(result, expected_models)


class TestOpenAIMixinCustomModelConstruction:
    """Test cases for mixed model types (LLM, embedding, rerank) through construct_model_from_identifier"""

    async def test_mixed_model_types_identification(self, mixin_with_custom_model_construction, mock_client_context):
        """Test that LLM, embedding, and rerank models are correctly identified with proper types and metadata"""
        # Create mock models: 1 embedding, 1 rerank, 1 LLM
        mock_embedding_model = MagicMock(id="text-embedding-3-small")
        mock_rerank_model = MagicMock(id="rerank-model-1")
        mock_llm_model = MagicMock(id="gpt-4")
        mock_models = [mock_embedding_model, mock_rerank_model, mock_llm_model]

        mock_client = MagicMock()

        async def mock_models_list():
            for model in mock_models:
                yield model

        mock_client.models.list.return_value = mock_models_list()

        with mock_client_context(mixin_with_custom_model_construction, mock_client):
            result = await mixin_with_custom_model_construction.list_models()

            assert result is not None
            assert len(result) == 3

            expected_models = {
                "text-embedding-3-small": {
                    "model_type": ModelType.embedding,
                    "metadata": {"embedding_dimension": 1536, "context_length": 8192},
                    "provider_id": "test-provider",
                    "provider_resource_id": "text-embedding-3-small",
                },
                "rerank-model-1": {
                    "model_type": ModelType.rerank,
                    "metadata": {},
                    "provider_id": "test-provider",
                    "provider_resource_id": "rerank-model-1",
                },
                "gpt-4": {
                    "model_type": ModelType.llm,
                    "metadata": {},
                    "provider_id": "test-provider",
                    "provider_resource_id": "gpt-4",
                },
            }

            _assert_models_match_expected(result, expected_models)


class TestOpenAIMixinAllowedModels:
    """Test cases for allowed_models filtering functionality"""

    async def test_list_models_with_allowed_models_filter(self, mixin, mock_client_with_models, mock_client_context):
        """Test that list_models filters models based on allowed_models"""
        mixin.config.allowed_models = ["some-mock-model-id", "another-mock-model-id"]

        with mock_client_context(mixin, mock_client_with_models):
            result = await mixin.list_models()

            assert result is not None
            assert len(result) == 2

            model_ids = [model.identifier for model in result]
            assert "some-mock-model-id" in model_ids
            assert "another-mock-model-id" in model_ids
            assert "final-mock-model-id" not in model_ids

    async def test_list_models_with_empty_allowed_models(self, mixin, mock_client_with_models, mock_client_context):
        """Test that empty allowed_models allows no models"""
        mixin.config.allowed_models = []

        with mock_client_context(mixin, mock_client_with_models):
            result = await mixin.list_models()

            assert result is not None
            assert len(result) == 0  # No models should be included

    async def test_list_models_with_omitted_allowed_models(self, mixin, mock_client_with_models, mock_client_context):
        """Test that omitted allowed_models allows all models"""
        assert mixin.config.allowed_models is None

        with mock_client_context(mixin, mock_client_with_models):
            result = await mixin.list_models()

            assert result is not None
            assert len(result) == 3  # All models should be included

            model_ids = [model.identifier for model in result]
            assert "some-mock-model-id" in model_ids
            assert "another-mock-model-id" in model_ids
            assert "final-mock-model-id" in model_ids

    async def test_check_model_availability_with_allowed_models(
        self, mixin, mock_client_with_models, mock_client_context
    ):
        """Test that check_model_availability respects allowed_models"""
        mixin.config.allowed_models = ["final-mock-model-id"]

        with mock_client_context(mixin, mock_client_with_models):
            assert await mixin.check_model_availability("final-mock-model-id")
            assert not await mixin.check_model_availability("some-mock-model-id")
            assert not await mixin.check_model_availability("another-mock-model-id")


class TestOpenAIMixinModelRegistration:
    """Test cases for model registration functionality"""

    async def test_register_model_success(self, mock_client_with_models, mock_client_context):
        """Test successful model registration when model is available"""
        config = RemoteInferenceProviderConfig()
        mixin = OpenAIMixinImpl(config=config)

        # Enable validation for this model
        model = Model(
            provider_id="test-provider",
            provider_resource_id="some-mock-model-id",
            identifier="test-model",
            model_type=ModelType.llm,
            model_validation=True,
        )

        with mock_client_context(mixin, mock_client_with_models):
            result = await mixin.register_model(model)

            assert result == model
            assert result.provider_id == "test-provider"
            assert result.provider_resource_id == "some-mock-model-id"
            assert result.identifier == "test-model"
            assert result.model_type == ModelType.llm
            mock_client_with_models.models.list.assert_called_once()

    async def test_register_model_not_available(self, mock_client_with_models, mock_client_context):
        """Test model registration failure when model is not available from provider"""
        config = RemoteInferenceProviderConfig()
        mixin = OpenAIMixinImpl(config=config)

        # Enable validation for this model
        model = Model(
            provider_id="test-provider",
            provider_resource_id="non-existent-model",
            identifier="test-model",
            model_type=ModelType.llm,
            model_validation=True,
        )

        with mock_client_context(mixin, mock_client_with_models):
            with pytest.raises(
                ValueError, match="Model non-existent-model is not available from provider test-provider"
            ):
                await mixin.register_model(model)
            mock_client_with_models.models.list.assert_called_once()

    async def test_register_model_with_allowed_models_filter(self, mock_client_with_models, mock_client_context):
        """Test model registration with allowed_models filtering"""
        config = RemoteInferenceProviderConfig(allowed_models=["some-mock-model-id"])
        mixin = OpenAIMixinImpl(config=config)

        # Test with allowed model (with validation enabled)
        allowed_model = Model(
            provider_id="test-provider",
            provider_resource_id="some-mock-model-id",
            identifier="allowed-model",
            model_type=ModelType.llm,
            model_validation=True,
        )

        # Test with disallowed model (with validation enabled)
        disallowed_model = Model(
            provider_id="test-provider",
            provider_resource_id="final-mock-model-id",
            identifier="disallowed-model",
            model_type=ModelType.llm,
            model_validation=True,
        )

        with mock_client_context(mixin, mock_client_with_models):
            result = await mixin.register_model(allowed_model)
            assert result == allowed_model
            with pytest.raises(
                ValueError, match="Model final-mock-model-id is not available from provider test-provider"
            ):
                await mixin.register_model(disallowed_model)
            mock_client_with_models.models.list.assert_called_once()

    async def test_register_embedding_model(self, mixin_with_embeddings, mock_client_context):
        """Test registration of embedding models with metadata"""
        mock_embedding_model = MagicMock(id="text-embedding-3-small")
        mock_models = [mock_embedding_model]

        mock_client = MagicMock()

        async def mock_models_list():
            for model in mock_models:
                yield model

        mock_client.models.list.return_value = mock_models_list()

        embedding_model = Model(
            provider_id="test-provider",
            provider_resource_id="text-embedding-3-small",
            identifier="embedding-test",
            model_type=ModelType.embedding,
        )

        with mock_client_context(mixin_with_embeddings, mock_client):
            result = await mixin_with_embeddings.register_model(embedding_model)
            assert result == embedding_model
            assert result.model_type == ModelType.embedding

    async def test_unregister_model(self, mixin):
        """Test model unregistration (should be no-op)"""
        # unregister_model should not raise any exceptions and return None
        result = await mixin.unregister_model("any-model-id")
        assert result is None

    async def test_should_refresh_models(self, mixin):
        """Test should_refresh_models method returns config value"""
        # Default config has refresh_models=False
        result = await mixin.should_refresh_models()
        assert result is False

        # With refresh_models=True, should return True
        config_with_refresh = RemoteInferenceProviderConfig(refresh_models=True)
        mixin_with_refresh = OpenAIMixinImpl(config=config_with_refresh)
        result_with_refresh = await mixin_with_refresh.should_refresh_models()
        assert result_with_refresh is True

    async def test_register_model_error_propagation(self, mock_client_with_exception, mock_client_context):
        """Test that errors from provider API are properly propagated during registration"""
        config = RemoteInferenceProviderConfig()
        mixin = OpenAIMixinImpl(config=config)

        # Enable validation for this model
        model = Model(
            provider_id="test-provider",
            provider_resource_id="some-model",
            identifier="test-model",
            model_type=ModelType.llm,
            model_validation=True,
        )

        with mock_client_context(mixin, mock_client_with_exception):
            # The exception from the API should be propagated
            with pytest.raises(Exception, match="API Error"):
                await mixin.register_model(model)

    async def test_register_model_default_behavior_no_validation(self, mock_client_with_models, mock_client_context):
        """Test model registration with default behavior (no validation)"""
        # Default behavior - no validation
        config = RemoteInferenceProviderConfig()
        mixin = OpenAIMixinImpl(config=config)

        model = Model(
            provider_id="test-provider",
            provider_resource_id="non-existent-model",
            identifier="test-model",
            model_type=ModelType.llm,
        )

        with mock_client_context(mixin, mock_client_with_models):
            # Should succeed without checking model availability (default behavior)
            result = await mixin.register_model(model)

            assert result == model
            # Verify that models.list() was NOT called
            mock_client_with_models.models.list.assert_not_called()

    async def test_register_model_with_validation_enabled(self, mock_client_with_models, mock_client_context):
        """Test that model-level model_validation=True enables validation"""
        # Default config (no provider-level validation setting)
        config = RemoteInferenceProviderConfig()
        mixin = OpenAIMixinImpl(config=config)

        # Model explicitly enables validation
        model = Model(
            provider_id="test-provider",
            provider_resource_id="non-existent-model",
            identifier="test-model",
            model_type=ModelType.llm,
            model_validation=True,
        )

        with mock_client_context(mixin, mock_client_with_models):
            # Should fail because model-level validation is enabled
            with pytest.raises(ValueError, match="Model non-existent-model is not available"):
                await mixin.register_model(model)
            # Verify that models.list() WAS called (validation happened)
            mock_client_with_models.models.list.assert_called_once()

    async def test_register_model_with_validation_explicitly_disabled(
        self, mock_client_with_models, mock_client_context
    ):
        """Test that model-level model_validation=False explicitly disables validation"""
        # Default config
        config = RemoteInferenceProviderConfig()
        mixin = OpenAIMixinImpl(config=config)

        # Model explicitly disables validation (though this is the default anyway)
        model = Model(
            provider_id="test-provider",
            provider_resource_id="non-existent-model",
            identifier="test-model",
            model_type=ModelType.llm,
            model_validation=False,
        )

        with mock_client_context(mixin, mock_client_with_models):
            # Should succeed because validation is disabled
            result = await mixin.register_model(model)

            assert result == model
            # Verify that models.list() was NOT called
            mock_client_with_models.models.list.assert_not_called()
