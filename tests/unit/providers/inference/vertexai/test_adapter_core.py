# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import SecretStr

from ogx.providers.remote.inference.vertexai.config import VertexAIConfig, VertexAIProviderDataValidator
from ogx.providers.remote.inference.vertexai.vertexai import VertexAIInferenceAdapter
from ogx_api import Model, ModelType

from .conftest import _async_pager


class TestVertexAIAdapterInit:
    def test_init_sets_config_and_default_client(
        self, adapter: VertexAIInferenceAdapter, vertex_config: VertexAIConfig
    ):
        """Test that init sets config and default client."""
        assert adapter.config == vertex_config
        assert adapter._default_client is None

    async def test_initialize_does_not_create_client(self, monkeypatch, adapter: VertexAIInferenceAdapter):
        """Test that initialize does NOT create default client (lazy initialization)."""
        client = object()

        monkeypatch.setattr(adapter, "_create_client", lambda **kwargs: client)

        await adapter.initialize()

        # With lazy initialization, client is NOT created during initialize()
        assert adapter._default_client is None

    async def test_initialize_does_not_fail_on_client_creation_error(
        self, monkeypatch, adapter: VertexAIInferenceAdapter
    ):
        """Test that initialize does not fail even if client creation would fail (lazy initialization)."""

        def _raise(**kwargs):
            """Raise a runtime error for failure-path testing."""
            raise RuntimeError("boom")

        monkeypatch.setattr(adapter, "_create_client", _raise)

        # Should not raise because client is not created during initialize()
        await adapter.initialize()

        assert adapter._default_client is None


class TestVertexAIClientManagement:
    def test_create_client_with_access_token_uses_credentials(self, monkeypatch):
        """Test that create client with access token uses credentials."""
        client_ctor = MagicMock(return_value=object())
        monkeypatch.setattr("ogx.providers.remote.inference.vertexai.vertexai.Client", client_ctor)

        adapter = VertexAIInferenceAdapter(config=VertexAIConfig(project="test-project", location="global"))
        client = adapter._create_client(
            project="test-project",
            location="global",
            access_token="token-123",
        )

        assert client is client_ctor.return_value
        kwargs = client_ctor.call_args.kwargs
        assert kwargs["vertexai"] is True
        assert kwargs["project"] == "test-project"
        assert kwargs["location"] == "global"
        assert kwargs["credentials"].token == "token-123"

    def test_get_client_uses_default_client(self, monkeypatch):
        """Test that get client uses default client."""
        adapter = VertexAIInferenceAdapter(config=VertexAIConfig(project="p", location="l"))
        default_client = object()
        adapter._default_client = cast(Any, default_client)
        monkeypatch.setattr(adapter, "_get_request_provider_overrides", lambda: None)

        assert adapter._get_client() is default_client

    def test_get_client_uses_provider_override_with_token(self, monkeypatch):
        """Test that get client uses provider override with token."""
        adapter = VertexAIInferenceAdapter(config=VertexAIConfig(project="p", location="l"))
        override = VertexAIProviderDataValidator(
            vertex_project="override-project",
            vertex_location="us-central1",
            vertex_access_token=SecretStr("override-token"),
        )
        monkeypatch.setattr(adapter, "_get_request_provider_overrides", lambda: override)

        create_client = MagicMock(return_value=object())
        monkeypatch.setattr(adapter, "_create_client", create_client)

        client = adapter._get_client()

        assert client is create_client.return_value
        create_client.assert_called_once_with(
            project="override-project",
            location="us-central1",
            access_token="override-token",
        )

    def test_get_client_project_override_reuses_configured_token(self, monkeypatch):
        """Test that get client project override reuses configured token."""
        adapter = VertexAIInferenceAdapter(
            config=VertexAIConfig(project="p", location="l", access_token=SecretStr("config-token")),
        )
        override = VertexAIProviderDataValidator(
            vertex_project="other-project",
            vertex_location=None,
            vertex_access_token=None,
        )
        monkeypatch.setattr(adapter, "_get_request_provider_overrides", lambda: override)

        create_client = MagicMock(return_value=object())
        monkeypatch.setattr(adapter, "_create_client", create_client)

        client = adapter._get_client()

        assert client is create_client.return_value
        create_client.assert_called_once_with(
            project="other-project",
            location="l",
            access_token="config-token",
        )

    def test_get_client_creates_default_client_lazily(self, monkeypatch):
        """Test that get client creates default client lazily when not available."""
        adapter = VertexAIInferenceAdapter(config=VertexAIConfig(project="p", location="l"))
        monkeypatch.setattr(adapter, "_get_request_provider_overrides", lambda: None)

        client = object()
        create_client = MagicMock(return_value=client)
        monkeypatch.setattr(adapter, "_create_client", create_client)

        # First call should create the client
        result = adapter._get_client()
        assert result is client
        assert adapter._default_client is client
        create_client.assert_called_once()

        # Second call should reuse the same client
        result2 = adapter._get_client()
        assert result2 is client
        assert create_client.call_count == 1  # Still only called once


class TestVertexAIModelListing:
    async def test_list_provider_model_ids_filters_and_deduplicates(self, monkeypatch):
        """Test that list provider model ids filters and deduplicates."""
        adapter = VertexAIInferenceAdapter(config=VertexAIConfig(project="p", location="l"))
        models = [
            SimpleNamespace(name="models/gemini-2.5-flash", supported_actions=["generateContent"]),
            SimpleNamespace(name="models/gemini-2.5-flash", supported_actions=["generateContent"]),
            SimpleNamespace(name="models/gemini-2.5-pro", supported_actions=[]),
            SimpleNamespace(name="models/text-embedding-004", supported_actions=["embedContent"]),
            SimpleNamespace(name="", supported_actions=["generateContent"]),
        ]

        async def fake_list(**kwargs):
            """Handle fake list."""
            return _async_pager(models)

        fake_client = SimpleNamespace(aio=SimpleNamespace(models=SimpleNamespace(list=fake_list)))
        monkeypatch.setattr(adapter, "_get_client", lambda: fake_client)

        result = await adapter.list_provider_model_ids()

        assert "models/gemini-2.5-flash" in result
        assert "models/gemini-2.5-pro" in result
        assert "models/text-embedding-004" in result
        assert "" not in result

    @pytest.mark.parametrize(
        "index,expected_id,expected_type,expected_metadata",
        [
            pytest.param(0, "models/gemini-2.5-flash", ModelType.llm, None, id="flash_llm"),
            pytest.param(1, "models/gemini-2.5-pro", ModelType.llm, None, id="pro_llm"),
            pytest.param(
                2,
                "models/gemini-embedding-001",
                ModelType.embedding,
                {"embedding_dimension": 3072, "context_length": 2048},
                id="embedding",
            ),
        ],
    )
    async def test_list_models_returns_correct_attributes(
        self, monkeypatch, adapter, index, expected_id, expected_type, expected_metadata
    ):
        """Test that list models returns correct attributes."""
        monkeypatch.setattr(
            adapter,
            "list_provider_model_ids",
            AsyncMock(return_value=["models/gemini-2.5-flash", "models/gemini-2.5-pro", "models/gemini-embedding-001"]),
        )

        models = await adapter.list_models()

        assert models is not None
        model = models[index]
        assert model.identifier == expected_id
        assert model.provider_resource_id == expected_id
        assert model.provider_id == "vertexai"
        assert model.model_type == expected_type
        if expected_metadata is not None:
            assert model.metadata == expected_metadata

    async def test_list_models_respects_allowed_models(self, monkeypatch, adapter: VertexAIInferenceAdapter):
        """Test that list models respects allowed models."""
        monkeypatch.setattr(adapter.config, "allowed_models", ["models/gemini-2.5-flash"])
        monkeypatch.setattr(
            adapter,
            "list_provider_model_ids",
            AsyncMock(return_value=["models/gemini-2.5-flash", "models/gemini-2.5-pro"]),
        )

        models = await adapter.list_models()

        assert models is not None
        assert len(models) == 1
        assert models[0].identifier == "models/gemini-2.5-flash"

    async def test_list_models_propagates_errors(self, monkeypatch):
        """Test that list models propagates errors."""
        adapter = VertexAIInferenceAdapter(config=VertexAIConfig(project="p", location="l"))
        monkeypatch.setattr(
            adapter,
            "list_provider_model_ids",
            AsyncMock(side_effect=RuntimeError("API unreachable")),
        )

        with pytest.raises(RuntimeError, match="API unreachable"):
            await adapter.list_models()

    async def test_should_refresh_models_returns_config_value(self):
        """Test that should refresh models returns config value."""
        adapter_default = VertexAIInferenceAdapter(config=VertexAIConfig(project="p", location="l"))
        assert await adapter_default.should_refresh_models() is False

        adapter_refresh = VertexAIInferenceAdapter(
            config=VertexAIConfig(project="p", location="l", refresh_models=True),
        )
        assert await adapter_refresh.should_refresh_models() is True


class TestVertexAIModelAvailability:
    @pytest.mark.parametrize(
        "model,available_models,error,expected",
        [
            ("models/gemini-2.5-flash", ["models/gemini-2.5-flash"], None, True),
            ("nonexistent-model", ["models/gemini-2.5-flash"], None, False),
            ("anything", None, RuntimeError("offline"), True),
        ],
    )
    async def test_check_model_availability(self, monkeypatch, model, available_models, error, expected):
        """Test that check model availability."""
        adapter = VertexAIInferenceAdapter(config=VertexAIConfig(project="p", location="l"))
        if error is not None:
            monkeypatch.setattr(adapter, "list_models", AsyncMock(side_effect=error))
        else:
            models = [
                Model(
                    provider_id="test-provider",
                    provider_resource_id=model_id,
                    identifier=model_id,
                    model_type=ModelType.llm,
                )
                for model_id in available_models
            ]

            async def mock_list_models():
                """Handle mock list models."""
                adapter._model_cache = {m.identifier: m for m in models}
                return models

            monkeypatch.setattr(adapter, "list_models", AsyncMock(side_effect=mock_list_models))

        assert await adapter.check_model_availability(model) is expected

    async def test_cache_hit_avoids_api_call(self, monkeypatch, adapter: VertexAIInferenceAdapter):
        """Verify lazy load + cache reuse prevents redundant API calls."""
        mock = AsyncMock(return_value=["models/gemini-2.5-flash"])
        monkeypatch.setattr(adapter, "list_provider_model_ids", mock)

        # First call populates cache (triggers list_models → list_provider_model_ids)
        result1 = await adapter.check_model_availability("models/gemini-2.5-flash")
        assert result1 is True

        # Second call uses cache — list_provider_model_ids NOT called again
        result2 = await adapter.check_model_availability("models/gemini-2.5-flash")
        assert result2 is True

        assert mock.call_count == 1  # API called exactly once

    async def test_cached_unknown_model_returns_false_without_api_call(
        self, monkeypatch, adapter: VertexAIInferenceAdapter
    ):
        """Once cache is populated, unknown model returns False without hitting API again."""
        mock = AsyncMock(return_value=["models/gemini-2.5-flash"])
        monkeypatch.setattr(adapter, "list_provider_model_ids", mock)

        # Populate the cache via first check
        await adapter.check_model_availability("models/gemini-2.5-flash")

        # Unknown model — must return False using cache, NOT re-call API
        result = await adapter.check_model_availability("nonexistent-model")
        assert result is False
        assert mock.call_count == 1  # Still only called once

    async def test_list_models_populates_model_cache(self, monkeypatch, adapter: VertexAIInferenceAdapter):
        """Verify list_models() populates _model_cache with Model objects."""
        monkeypatch.setattr(
            adapter,
            "list_provider_model_ids",
            AsyncMock(return_value=["models/gemini-2.5-flash", "models/gemini-2.5-pro"]),
        )

        await adapter.list_models()

        assert "models/gemini-2.5-flash" in adapter._model_cache
        assert "models/gemini-2.5-pro" in adapter._model_cache
        assert len(adapter._model_cache) == 2

    async def test_list_models_clears_cache_on_refresh(self, monkeypatch, adapter: VertexAIInferenceAdapter):
        """Verify cache is cleared and repopulated on list_models() call."""
        mock = AsyncMock(return_value=["models/model-a"])
        monkeypatch.setattr(adapter, "list_provider_model_ids", mock)

        await adapter.list_models()
        assert "models/model-a" in adapter._model_cache

        # Change what the API returns
        mock.return_value = ["models/model-b"]
        await adapter.list_models()

        # Cache should now only have model-b, not model-a
        assert "models/model-b" in adapter._model_cache
        assert "models/model-a" not in adapter._model_cache
        assert len(adapter._model_cache) == 1


class TestVertexAIAllowedModelsValidation:
    def test_validate_allowed_model_passes(self):
        """Test that validate allowed model passes."""
        adapter = VertexAIInferenceAdapter(
            config=VertexAIConfig(project="p", location="l", allowed_models=["models/gemini-2.5-flash"]),
        )
        adapter._validate_model_allowed("models/gemini-2.5-flash")

    def test_validate_disallowed_model_raises(self):
        """Test that validate disallowed model raises."""
        adapter = VertexAIInferenceAdapter(
            config=VertexAIConfig(project="p", location="l", allowed_models=["models/gemini-2.5-flash"]),
        )
        with pytest.raises(ValueError, match="not in the allowed models list"):
            adapter._validate_model_allowed("models/gemini-2.5-pro")

    def test_validate_no_allowed_models_passes_anything(self):
        """Test that validate no allowed models passes anything."""
        adapter = VertexAIInferenceAdapter(config=VertexAIConfig(project="p", location="l"))
        adapter._validate_model_allowed("any-model-at-all")


class TestVertexAIUnsupportedOps:
    @pytest.mark.parametrize(
        "method_name,call_kwargs,error_pattern",
        [
            ("rerank", {"request": cast(Any, None)}, "rerank not yet implemented"),
        ],
    )
    async def test_unsupported_operations_raise(self, method_name, call_kwargs, error_pattern):
        """Verify that unimplemented operations raise NotImplementedError."""
        adapter = VertexAIInferenceAdapter(config=VertexAIConfig(project="p", location="l"))

        with pytest.raises(NotImplementedError, match=error_pattern):
            await getattr(adapter, method_name)(**call_kwargs)
