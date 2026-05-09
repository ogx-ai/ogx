# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""
Tests for ABAC enforcement in vector store operations.

This test suite verifies that all vector store operations properly enforce
authorization checks through the router -> routing table -> ABAC flow.
"""

from unittest.mock import AsyncMock, Mock

import pytest

from ogx.core.routers.vector_io import VectorIORouter
from ogx.core.routing_tables.vector_stores import VectorStoresRoutingTable
from ogx_api import (
    ChunkMetadata,
    EmbeddedChunk,
    InsertChunksRequest,
    OpenAIAttachFileRequest,
    OpenAICreateVectorStoreFileBatchRequestWithExtraBody,
    OpenAISearchVectorStoreRequest,
    OpenAIUpdateVectorStoreFileRequest,
    OpenAIUpdateVectorStoreRequest,
    QueryChunksRequest,
    QueryChunksResponse,
    VectorStoreChunkingStrategyStatic,
    VectorStoreChunkingStrategyStaticConfig,
    VectorStoreDeleteResponse,
    VectorStoreFileBatchObject,
    VectorStoreFileCounts,
    VectorStoreFileDeleteResponse,
    VectorStoreFileObject,
    VectorStoreFilesListInBatchResponse,
    VectorStoreListFilesResponse,
    VectorStoreObject,
    VectorStoreSearchResponsePage,
)


class MockDistRegistry:
    """Mock distribution registry for testing."""

    def __init__(self):
        self.dist = None

    def get_cached(self, type_name: str, identifier: str):
        return None


@pytest.fixture
def mock_provider():
    """Create a mock provider that returns valid responses for all operations."""
    provider = Mock()

    provider.insert_chunks = AsyncMock()
    provider.query_chunks = AsyncMock(return_value=QueryChunksResponse(chunks=[], scores=[]))
    provider.openai_retrieve_vector_store = AsyncMock(
        return_value=VectorStoreObject(
            id="vs_123",
            created_at=1234567890,
            name="test",
            usage_bytes=0,
            status="completed",
            file_counts=VectorStoreFileCounts(completed=0, cancelled=0, failed=0, in_progress=0, total=0),
        )
    )
    provider.openai_update_vector_store = AsyncMock(
        return_value=VectorStoreObject(
            id="vs_123",
            created_at=1234567890,
            name="test",
            usage_bytes=0,
            status="completed",
            file_counts=VectorStoreFileCounts(completed=0, cancelled=0, failed=0, in_progress=0, total=0),
        )
    )
    provider.openai_delete_vector_store = AsyncMock(return_value=VectorStoreDeleteResponse(id="vs_123", deleted=True))
    provider.openai_search_vector_store = AsyncMock(
        return_value=VectorStoreSearchResponsePage(search_query=["test"], data=[], has_more=False)
    )
    provider.openai_attach_file_to_vector_store = AsyncMock(
        return_value=VectorStoreFileObject(
            id="file_123",
            chunking_strategy=VectorStoreChunkingStrategyStatic(
                static=VectorStoreChunkingStrategyStaticConfig(max_chunk_size_tokens=800, chunk_overlap_tokens=400)
            ),
            created_at=1234567890,
            status="completed",
            usage_bytes=0,
            vector_store_id="vs_123",
        )
    )
    provider.openai_list_files_in_vector_store = AsyncMock(
        return_value=VectorStoreListFilesResponse(data=[], first_id="", last_id="", has_more=False)
    )
    provider.openai_retrieve_vector_store_file = AsyncMock(
        return_value=VectorStoreFileObject(
            id="file_123",
            chunking_strategy=VectorStoreChunkingStrategyStatic(
                static=VectorStoreChunkingStrategyStaticConfig(max_chunk_size_tokens=800, chunk_overlap_tokens=400)
            ),
            created_at=1234567890,
            status="completed",
            usage_bytes=0,
            vector_store_id="vs_123",
        )
    )
    provider.openai_update_vector_store_file = AsyncMock(
        return_value=VectorStoreFileObject(
            id="file_123",
            chunking_strategy=VectorStoreChunkingStrategyStatic(
                static=VectorStoreChunkingStrategyStaticConfig(max_chunk_size_tokens=800, chunk_overlap_tokens=400)
            ),
            created_at=1234567890,
            status="completed",
            usage_bytes=0,
            vector_store_id="vs_123",
        )
    )
    provider.openai_delete_vector_store_file = AsyncMock(
        return_value=VectorStoreFileDeleteResponse(id="file_123", deleted=True)
    )
    provider.openai_create_vector_store_file_batch = AsyncMock(
        return_value=VectorStoreFileBatchObject(
            id="batch_123",
            created_at=1234567890,
            vector_store_id="vs_123",
            status="in_progress",
            file_counts=VectorStoreFileCounts(completed=0, cancelled=0, failed=0, in_progress=2, total=2),
        )
    )
    provider.openai_retrieve_vector_store_file_batch = AsyncMock(
        return_value=VectorStoreFileBatchObject(
            id="batch_123",
            created_at=1234567890,
            vector_store_id="vs_123",
            status="completed",
            file_counts=VectorStoreFileCounts(completed=2, cancelled=0, failed=0, in_progress=0, total=2),
        )
    )
    provider.openai_list_files_in_vector_store_file_batch = AsyncMock(
        return_value=VectorStoreFilesListInBatchResponse(data=[], first_id="", last_id="", has_more=False)
    )
    provider.openai_cancel_vector_store_file_batch = AsyncMock(
        return_value=VectorStoreFileBatchObject(
            id="batch_123",
            created_at=1234567890,
            vector_store_id="vs_123",
            status="cancelled",
            file_counts=VectorStoreFileCounts(completed=0, cancelled=2, failed=0, in_progress=0, total=2),
        )
    )

    return provider


@pytest.fixture
def router_with_real_routing_table(mock_provider):
    """Create router with real routing table for integration testing."""
    mock_dist_registry = MockDistRegistry()

    routing_table = VectorStoresRoutingTable(
        impls_by_provider_id={"test-provider": mock_provider},
        dist_registry=mock_dist_registry,
        policy=[],
    )

    # Mock get_provider_impl to return our mock provider
    routing_table.get_provider_impl = AsyncMock(return_value=mock_provider)

    # Mock get_object_by_identifier to return a mock vector store object
    # This is needed by assert_action_allowed to check permissions
    from ogx.core.datatypes import VectorStoreWithOwner
    from ogx_api import ResourceType

    mock_vector_store = VectorStoreWithOwner(
        identifier="vs_123",
        provider_id="test-provider",
        provider_resource_id="vs_123",
        type=ResourceType.vector_store,
        embedding_model="test-model",
        embedding_dimension=768,
    )
    routing_table.get_object_by_identifier = AsyncMock(return_value=mock_vector_store)

    # Spy on assert_action_allowed to verify it's called correctly
    original_assert = routing_table.assert_action_allowed
    routing_table.assert_action_allowed = AsyncMock(wraps=original_assert)

    # Create router with real routing table
    router = VectorIORouter(routing_table)

    return router, routing_table, mock_provider


@pytest.mark.parametrize(
    "operation_name,expected_action,router_call,provider_method",
    [
        (
            "insert_chunks",
            "update",
            lambda r: r.insert_chunks(
                InsertChunksRequest(
                    vector_store_id="vs_123",
                    chunks=[
                        EmbeddedChunk(
                            content="test",
                            chunk_id="c1",
                            metadata=dict(
                                document_id="test_doc",
                                chunk_id="c1",
                                created_timestamp=1234567890,
                                updated_timestamp=1234567890,
                                chunk_embedding_model="test-model",
                                chunk_embedding_dimension=768,
                                content_token_count=1,
                            ),
                            chunk_metadata=ChunkMetadata(
                                document_id="test_doc",
                                chunk_id="c1",
                                created_timestamp=1234567890,
                                updated_timestamp=1234567890,
                                content_token_count=1,
                            ),
                            embedding=[0.1] * 768,
                            embedding_model="test-model",
                            embedding_dimension=768,
                        )
                    ],
                )
            ),
            "insert_chunks",
        ),
        (
            "query_chunks",
            "read",
            lambda r: r.query_chunks(QueryChunksRequest(vector_store_id="vs_123", query="test")),
            "query_chunks",
        ),
        (
            "openai_retrieve_vector_store",
            "read",
            lambda r: r.openai_retrieve_vector_store("vs_123"),
            "openai_retrieve_vector_store",
        ),
        (
            "openai_update_vector_store",
            "update",
            lambda r: r.openai_update_vector_store("vs_123", request=OpenAIUpdateVectorStoreRequest(name="test")),
            "openai_update_vector_store",
        ),
        (
            "openai_delete_vector_store",
            "delete",
            lambda r: r.openai_delete_vector_store("vs_123"),
            "openai_delete_vector_store",
        ),
        (
            "openai_search_vector_store",
            "read",
            lambda r: r.openai_search_vector_store("vs_123", request=OpenAISearchVectorStoreRequest(query="test")),
            "openai_search_vector_store",
        ),
        (
            "openai_attach_file_to_vector_store",
            "update",
            lambda r: r.openai_attach_file_to_vector_store(
                "vs_123", request=OpenAIAttachFileRequest(file_id="file_123")
            ),
            "openai_attach_file_to_vector_store",
        ),
        (
            "openai_list_files_in_vector_store",
            "read",
            lambda r: r.openai_list_files_in_vector_store("vs_123"),
            "openai_list_files_in_vector_store",
        ),
        (
            "openai_retrieve_vector_store_file",
            "read",
            lambda r: r.openai_retrieve_vector_store_file("vs_123", "file_123"),
            "openai_retrieve_vector_store_file",
        ),
        (
            "openai_update_vector_store_file",
            "update",
            lambda r: r.openai_update_vector_store_file(
                "vs_123",
                "file_123",
                request=OpenAIUpdateVectorStoreFileRequest(attributes={}),
            ),
            "openai_update_vector_store_file",
        ),
        (
            "openai_delete_vector_store_file",
            "delete",
            lambda r: r.openai_delete_vector_store_file("vs_123", "file_123"),
            "openai_delete_vector_store_file",
        ),
        (
            "openai_create_vector_store_file_batch",
            "update",
            lambda r: r.openai_create_vector_store_file_batch(
                "vs_123", OpenAICreateVectorStoreFileBatchRequestWithExtraBody(file_ids=["f1"])
            ),
            "openai_create_vector_store_file_batch",
        ),
        (
            "openai_retrieve_vector_store_file_batch",
            "read",
            lambda r: r.openai_retrieve_vector_store_file_batch("batch_123", "vs_123"),
            "openai_retrieve_vector_store_file_batch",
        ),
        (
            "openai_list_files_in_vector_store_file_batch",
            "read",
            lambda r: r.openai_list_files_in_vector_store_file_batch("batch_123", "vs_123"),
            "openai_list_files_in_vector_store_file_batch",
        ),
        (
            "openai_cancel_vector_store_file_batch",
            "update",
            lambda r: r.openai_cancel_vector_store_file_batch("batch_123", "vs_123"),
            "openai_cancel_vector_store_file_batch",
        ),
    ],
)
async def test_operation_enforces_correct_abac_permission(
    operation_name, expected_action, router_call, provider_method, router_with_real_routing_table
):
    """Test that each operation flows through router -> routing table -> ABAC check -> provider.

    This verifies:
    1. Router delegates to routing table (not directly to provider)
    2. Routing table enforces ABAC with correct action
    3. Provider is called only after authorization succeeds
    """
    router, routing_table, mock_provider = router_with_real_routing_table

    # Execute the operation through the router
    await router_call(router)

    # Verify ABAC check happened with correct action and resource
    routing_table.assert_action_allowed.assert_called_once_with(expected_action, "vector_store", "vs_123")

    # Verify provider was called after authorization
    provider_mock = getattr(mock_provider, provider_method)
    provider_mock.assert_called_once()


async def test_operations_fail_before_provider_when_unauthorized(router_with_real_routing_table):
    """Test that all operations fail at ABAC check before calling provider when unauthorized."""
    router, routing_table, mock_provider = router_with_real_routing_table

    # Make assert_action_allowed raise PermissionError
    routing_table.assert_action_allowed.side_effect = PermissionError("Access denied")

    # Test all operations fail before reaching provider
    operations = [
        (
            "insert_chunks",
            lambda: router.insert_chunks(
                InsertChunksRequest(
                    vector_store_id="vs_123",
                    chunks=[
                        EmbeddedChunk(
                            content="test",
                            chunk_id="c1",
                            metadata=dict(
                                document_id="test_doc",
                                chunk_id="c1",
                                created_timestamp=1234567890,
                                updated_timestamp=1234567890,
                                chunk_embedding_model="test-model",
                                chunk_embedding_dimension=768,
                                content_token_count=1,
                            ),
                            chunk_metadata=ChunkMetadata(
                                document_id="test_doc",
                                chunk_id="c1",
                                created_timestamp=1234567890,
                                updated_timestamp=1234567890,
                                content_token_count=1,
                            ),
                            embedding=[0.1] * 768,
                            embedding_model="test-model",
                            embedding_dimension=768,
                        )
                    ],
                )
            ),
        ),
        ("query_chunks", lambda: router.query_chunks(QueryChunksRequest(vector_store_id="vs_123", query="test"))),
        ("openai_retrieve_vector_store", lambda: router.openai_retrieve_vector_store("vs_123")),
        (
            "openai_update_vector_store",
            lambda: router.openai_update_vector_store("vs_123", request=OpenAIUpdateVectorStoreRequest(name="test")),
        ),
        ("openai_delete_vector_store", lambda: router.openai_delete_vector_store("vs_123")),
        (
            "openai_search_vector_store",
            lambda: router.openai_search_vector_store("vs_123", request=OpenAISearchVectorStoreRequest(query="test")),
        ),
        (
            "openai_attach_file_to_vector_store",
            lambda: router.openai_attach_file_to_vector_store(
                "vs_123", request=OpenAIAttachFileRequest(file_id="file_123")
            ),
        ),
        ("openai_list_files_in_vector_store", lambda: router.openai_list_files_in_vector_store("vs_123")),
        (
            "openai_retrieve_vector_store_file",
            lambda: router.openai_retrieve_vector_store_file("vs_123", "file_123"),
        ),
        (
            "openai_update_vector_store_file",
            lambda: router.openai_update_vector_store_file(
                "vs_123", "file_123", request=OpenAIUpdateVectorStoreFileRequest(attributes={})
            ),
        ),
        (
            "openai_delete_vector_store_file",
            lambda: router.openai_delete_vector_store_file("vs_123", "file_123"),
        ),
        (
            "openai_create_vector_store_file_batch",
            lambda: router.openai_create_vector_store_file_batch(
                "vs_123", OpenAICreateVectorStoreFileBatchRequestWithExtraBody(file_ids=["f1"])
            ),
        ),
        (
            "openai_retrieve_vector_store_file_batch",
            lambda: router.openai_retrieve_vector_store_file_batch("batch_123", "vs_123"),
        ),
        (
            "openai_list_files_in_vector_store_file_batch",
            lambda: router.openai_list_files_in_vector_store_file_batch("batch_123", "vs_123"),
        ),
        (
            "openai_cancel_vector_store_file_batch",
            lambda: router.openai_cancel_vector_store_file_batch("batch_123", "vs_123"),
        ),
    ]

    for op_name, op_func in operations:
        # Reset mocks
        routing_table.assert_action_allowed.reset_mock()
        routing_table.assert_action_allowed.side_effect = PermissionError("Access denied")

        # Operation should fail with PermissionError
        with pytest.raises(PermissionError, match="Access denied"):
            await op_func()

        # Verify ABAC check was called
        assert routing_table.assert_action_allowed.called, f"{op_name} should check permissions"

    # Verify provider was NEVER called (all 15 operations)
    mock_provider.insert_chunks.assert_not_called()
    mock_provider.query_chunks.assert_not_called()
    mock_provider.openai_retrieve_vector_store.assert_not_called()
    mock_provider.openai_update_vector_store.assert_not_called()
    mock_provider.openai_delete_vector_store.assert_not_called()
    mock_provider.openai_search_vector_store.assert_not_called()
    mock_provider.openai_attach_file_to_vector_store.assert_not_called()
    mock_provider.openai_list_files_in_vector_store.assert_not_called()
    mock_provider.openai_retrieve_vector_store_file.assert_not_called()
    mock_provider.openai_update_vector_store_file.assert_not_called()
    mock_provider.openai_delete_vector_store_file.assert_not_called()
    mock_provider.openai_create_vector_store_file_batch.assert_not_called()
    mock_provider.openai_retrieve_vector_store_file_batch.assert_not_called()
    mock_provider.openai_list_files_in_vector_store_file_batch.assert_not_called()
    mock_provider.openai_cancel_vector_store_file_batch.assert_not_called()
