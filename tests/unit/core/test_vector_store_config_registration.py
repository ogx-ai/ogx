# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Tests for config-based vector store registration."""

import json
from unittest.mock import AsyncMock

from ogx.core.datatypes import RegisteredResources, StackConfig, VectorStoresConfig
from ogx.core.routing_tables.vector_stores import VectorStoresRoutingTable
from ogx.core.stack import register_resources
from ogx.core.storage.datatypes import ServerStoresConfig, StorageConfig
from ogx.providers.utils.memory.openai_vector_store_mixin import OpenAIVectorStoreMixin
from ogx_api import Api, Model, ModelType, VectorStore
from ogx_api.vector_stores import VectorStoreInput


class TestVectorStoreRegistration:
    """Test vector store registration from configuration."""

    async def test_basic_registration(self):
        """Test that vector stores can be registered from config."""

        # mock models API which returns an embedding model
        class MockModelsAPI:
            async def get(self, identifier: str):
                if "embedding" in identifier:
                    return type(
                        "Model",
                        (),
                        {
                            "identifier": identifier,
                            "model_type": ModelType.embedding,
                            "embedding_dimension": 768,
                        },
                    )()
                return None

            async def register_model(self, **kwargs):
                """Mock register_model"""
                pass

            async def list_models(self):
                """Mock list_models, returns empty list."""
                return []

        # mock vector_stores routing table
        class MockVectorStoresRoutingTable:
            def __init__(self):
                self.registered = []

            async def register_vector_store(
                self,
                vector_store_id: str,
                embedding_model: str,
                embedding_dimension: int,
                provider_id: str | None = None,
                provider_vector_store_id: str | None = None,
                vector_store_name: str | None = None,
            ):
                self.registered.append(
                    {
                        "vector_store_id": vector_store_id,
                        "embedding_model": embedding_model,
                        "embedding_dimension": embedding_dimension,
                        "provider_id": provider_id,
                    }
                )
                return type(
                    "VectorStore",
                    (),
                    {
                        "identifier": vector_store_id,
                        "embedding_model": embedding_model,
                        "embedding_dimension": embedding_dimension,
                        "provider_id": provider_id,
                    },
                )()

            async def list_vector_stores(self):
                """Mock list_vector_stores, returns all registered stores."""
                return [
                    type(
                        "VectorStore",
                        (),
                        {
                            "identifier": reg["vector_store_id"],
                            "provider_id": reg["provider_id"],
                            "embedding_model": reg["embedding_model"],
                            "embedding_dimension": reg["embedding_dimension"],
                        },
                    )()
                    for reg in self.registered
                ]

        # config with vector stores
        run_config = StackConfig(
            image_name="test",
            providers={},
            storage=StorageConfig(
                backends={},
                stores=ServerStoresConfig(
                    metadata=None,
                    inference=None,
                    conversations=None,
                    prompts=None,
                ),
            ),
            vector_stores=VectorStoresConfig(
                default_provider_id="test_provider",
            ),
            registered_resources=RegisteredResources(
                vector_stores=[
                    VectorStoreInput(
                        vector_store_id="test_store_1",
                        embedding_model="test/embedding-model",
                        embedding_dimension=768,
                        provider_id="test_provider",
                    ),
                    VectorStoreInput(
                        vector_store_id="test_store_2",
                        embedding_model="test/embedding-model",
                        embedding_dimension=384,
                        provider_id="test_provider",
                        vector_store_name="My Test Store",
                    ),
                ],
            ),
        )

        mock_vector_stores_api = MockVectorStoresRoutingTable()
        impls = {
            Api.models: MockModelsAPI(),
            Api.vector_stores: mock_vector_stores_api,
        }

        await register_resources(run_config, impls)

        assert len(mock_vector_stores_api.registered) == 2

        # Verify first vector store
        assert mock_vector_stores_api.registered[0]["vector_store_id"] == "test_store_1"
        assert mock_vector_stores_api.registered[0]["embedding_model"] == "test/embedding-model"
        assert mock_vector_stores_api.registered[0]["embedding_dimension"] == 768
        assert mock_vector_stores_api.registered[0]["provider_id"] == "test_provider"

        # Verify second vector store
        assert mock_vector_stores_api.registered[1]["vector_store_id"] == "test_store_2"
        assert mock_vector_stores_api.registered[1]["embedding_dimension"] == 384

    async def test_empty_config(self):
        """Test that empty vector_stores config doesn't cause errors."""
        run_config = StackConfig(
            image_name="test",
            providers={},
            storage=StorageConfig(
                backends={},
                stores=ServerStoresConfig(
                    metadata=None,
                    inference=None,
                    conversations=None,
                    prompts=None,
                ),
            ),
            registered_resources=RegisteredResources(
                vector_stores=[],  # Empty list
            ),
        )

        impls = {}

        # Should not raise any errors
        await register_resources(run_config, impls)

    async def test_registration_with_optional_fields(self):
        """Test vector store registration with all optional fields."""

        class MockModelsAPI:
            async def get(self, identifier: str):
                return type(
                    "Model",
                    (),
                    {
                        "identifier": identifier,
                        "model_type": ModelType.embedding,
                        "embedding_dimension": 512,
                    },
                )()

            async def register_model(self, **kwargs):
                """Mock register_model"""
                pass

            async def list_models(self):
                """Mock list_models, returns empty list."""
                return []

        class MockVectorStoresRoutingTable:
            def __init__(self):
                self.last_registered = None
                self.all_registered = []

            async def register_vector_store(self, **kwargs):
                self.last_registered = kwargs
                self.all_registered.append(kwargs)
                return type("VectorStore", (), kwargs)()

            async def list_vector_stores(self):
                """Mock list_vector_stores, returns all registered stores."""
                return [
                    type(
                        "VectorStore",
                        (),
                        {
                            "identifier": reg.get("vector_store_id"),
                            "provider_id": reg.get("provider_id"),
                            "embedding_model": reg.get("embedding_model"),
                            "embedding_dimension": reg.get("embedding_dimension"),
                        },
                    )()
                    for reg in self.all_registered
                ]

        run_config = StackConfig(
            image_name="test",
            providers={},
            storage=StorageConfig(
                backends={},
                stores=ServerStoresConfig(
                    metadata=None,
                    inference=None,
                    conversations=None,
                    prompts=None,
                ),
            ),
            registered_resources=RegisteredResources(
                vector_stores=[
                    VectorStoreInput(
                        vector_store_id="full_store",
                        embedding_model="test/model",
                        embedding_dimension=512,
                        provider_id="my_provider",
                        provider_vector_store_id="custom_id",
                        vector_store_name="Full Featured Store",
                    ),
                ],
            ),
        )

        mock_api = MockVectorStoresRoutingTable()
        impls = {
            Api.models: MockModelsAPI(),
            Api.vector_stores: mock_api,
        }

        await register_resources(run_config, impls)

        assert mock_api.last_registered["vector_store_id"] == "full_store"
        assert mock_api.last_registered["provider_vector_store_id"] == "custom_id"
        assert mock_api.last_registered["vector_store_name"] == "Full Featured Store"


class TestOpenAIMetadataCreation:
    """Test OpenAI-compatible metadata creation for vector stores."""

    async def test_config_registration_creates_openai_metadata(self, disk_dist_registry, sqlite_kvstore):
        """Test that registering vector stores from config creates OpenAI metadata when provider supports it."""

        class MockVectorIOProvider(OpenAIVectorStoreMixin):
            __provider_spec__ = type("ProviderSpec", (), {"api": Api.vector_io})

            def __init__(self, kvstore):
                mock_inference_api = AsyncMock()
                super().__init__(
                    inference_api=mock_inference_api,
                    kvstore=kvstore,
                )

            async def register_vector_store(self, vector_store: VectorStore):
                return vector_store

            async def unregister_vector_store(self, vector_store_id: str):
                pass

            async def insert_chunks(self, vector_store_id: str, chunks, **kwargs):
                pass

            async def query_chunks(self, vector_store_id: str, query, **kwargs):
                pass

            async def delete_chunks(self, request):
                pass

        class MockModelsAPI:
            async def get(self, identifier: str):
                if "embedding" in identifier:
                    return type(
                        "Model",
                        (),
                        {
                            "identifier": identifier,
                            "model_type": ModelType.embedding,
                            "embedding_dimension": 768,
                        },
                    )()
                return None

            async def register_model(self, **kwargs):
                pass

            async def list_models(self):
                return []

        test_model = Model(
            identifier="test/embedding-model",
            model_type=ModelType.embedding,
            provider_id="test_provider",
            provider_resource_id="test/embedding-model",
        )
        await disk_dist_registry.register(test_model)

        mock_provider = MockVectorIOProvider(sqlite_kvstore)
        impls_by_provider_id = {"test_provider": mock_provider}
        policy = []

        routing_table = VectorStoresRoutingTable(
            impls_by_provider_id=impls_by_provider_id,
            dist_registry=disk_dist_registry,
            policy=policy,
        )

        # config-based registration
        run_config = StackConfig(
            image_name="test",
            providers={},
            storage=StorageConfig(
                backends={},
                stores=ServerStoresConfig(
                    metadata=None,
                    inference=None,
                    conversations=None,
                    prompts=None,
                ),
            ),
            vector_stores=VectorStoresConfig(
                default_provider_id="test_provider",
            ),
            registered_resources=RegisteredResources(
                vector_stores=[
                    VectorStoreInput(
                        vector_store_id="openai_store",
                        embedding_model="test/embedding-model",
                        embedding_dimension=768,
                        provider_id="test_provider",
                        vector_store_name="OpenAI Compatible Store",
                    ),
                ],
            ),
        )

        impls = {
            Api.models: MockModelsAPI(),
            Api.vector_stores: routing_table,
        }

        await register_resources(run_config, impls)

        # Verify that metadata was actually stored in kvstore
        stored_key = "openai_vector_stores:v3::openai_store"
        stored_value = await sqlite_kvstore.get(stored_key)
        assert stored_value is not None, "OpenAI metadata should be stored in kvstore"

        # Parse and verify the stored metadata matches OpenAI spec
        stored_metadata = json.loads(stored_value)
        assert stored_metadata["id"] == "openai_store"
        assert stored_metadata["object"] == "vector_store"
        assert stored_metadata["name"] == "OpenAI Compatible Store"
        assert stored_metadata["status"] == "completed"
        assert "created_at" in stored_metadata
        assert "last_active_at" in stored_metadata
        assert stored_metadata["usage_bytes"] == 0
        assert stored_metadata["file_counts"]["total"] == 0
        assert stored_metadata["metadata"]["provider_id"] == "test_provider"
        assert "provider_vector_store_id" in stored_metadata["metadata"]
        assert stored_metadata["metadata"]["embedding_model"] == "test/embedding-model"
        assert stored_metadata["metadata"]["embedding_dimension"] == "768"

        # Verify it's also in the provider's memory cache
        assert "openai_store" in mock_provider.openai_vector_stores
        cached_metadata = mock_provider.openai_vector_stores["openai_store"]
        assert cached_metadata["id"] == "openai_store"
        assert cached_metadata["name"] == "OpenAI Compatible Store"
        assert cached_metadata["status"] == "completed"
        assert cached_metadata["metadata"]["embedding_model"] == "test/embedding-model"
        assert cached_metadata["metadata"]["embedding_dimension"] == "768"
