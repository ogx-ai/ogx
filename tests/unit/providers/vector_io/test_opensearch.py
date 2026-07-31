# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from ogx.providers.remote.vector_io.opensearch.config import (
    OpenSearchVectorIOConfig,
)
from ogx.providers.remote.vector_io.opensearch.opensearch import (
    OpenSearchVectorIOAdapter,
)
from ogx_api import (
    EmbeddedChunk,
    InsertChunksRequest,
    QueryChunksRequest,
    QueryChunksResponse,
    VectorStore,
)


def make_chunk(**overrides):
    kw = dict(
        content="test content",
        chunk_id="chunk1",
        metadata={"key": "value"},
        chunk_metadata={},
        embedding=[0.1, 0.2, 0.3, 0.4],
        embedding_model="test_model",
        embedding_dimension=4,
    )
    kw.update(overrides)
    return EmbeddedChunk(**kw)


class TestOpenSearchVectorIOAdapter(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.config = OpenSearchVectorIOConfig(
            host="localhost",
            port=9200,
        )
        self.inference_api = MagicMock()
        self.files_api = MagicMock()

    @patch("ogx.providers.remote.vector_io.opensearch.opensearch.OpenSearch")
    async def test_initialize(self, mock_opensearch):
        adapter = OpenSearchVectorIOAdapter(self.config, self.inference_api, self.files_api)

        # Mock OpenSearch client
        mock_client = MagicMock()
        mock_opensearch.return_value = mock_client
        mock_client.info.return_value = {"version": {"number": "2.11.0"}}

        # Mock kvstore and openai vector stores to avoid persistence requirements
        adapter.kvstore = AsyncMock()
        with patch.object(adapter, "initialize_openai_vector_stores", new_callable=AsyncMock):
            await adapter.initialize()

        mock_opensearch.assert_called_once()
        # Verify hosts config usage
        call_args = mock_opensearch.call_args[1]
        self.assertEqual(call_args["hosts"], [{"host": "localhost", "port": 9200}])

    @patch("ogx.providers.remote.vector_io.opensearch.opensearch.OpenSearch")
    async def test_register_vector_store(self, mock_opensearch):
        adapter = OpenSearchVectorIOAdapter(self.config, self.inference_api, self.files_api, file_processor_api=None)
        mock_client = MagicMock()
        mock_opensearch.return_value = mock_client
        adapter.kvstore = AsyncMock()
        with patch.object(adapter, "initialize_openai_vector_stores", new_callable=AsyncMock):
            await adapter.initialize()

        vector_store = VectorStore(
            identifier="test_store",
            provider_id="test_provider",
            provider_resource_id="test_resource",
            embedding_dimension=384,
            embedding_model="test_model",
        )

        mock_client.indices.exists.return_value = False

        await adapter.register_vector_store(vector_store)

        # Verify index creation
        mock_client.indices.create.assert_called_once()
        call_args = mock_client.indices.create.call_args[1]
        self.assertEqual(call_args["index"], "test_store")
        self.assertTrue("knn" in call_args["body"]["settings"]["index"])

    @patch("ogx.providers.remote.vector_io.opensearch.opensearch.OpenSearch")
    @patch("ogx.providers.remote.vector_io.opensearch.opensearch.helpers.bulk")
    async def test_insert_chunks(self, mock_bulk, mock_opensearch):
        adapter = OpenSearchVectorIOAdapter(self.config, self.inference_api, self.files_api, file_processor_api=None)
        mock_client = MagicMock()
        mock_opensearch.return_value = mock_client
        mock_client.indices.exists.return_value = True
        adapter.kvstore = AsyncMock()
        with patch.object(adapter, "initialize_openai_vector_stores", new_callable=AsyncMock):
            await adapter.initialize()

        vector_store = VectorStore(
            identifier="test_store",
            provider_id="test_provider",
            provider_resource_id="test_resource",
            embedding_dimension=4,
            embedding_model="test_model",
        )
        await adapter.register_vector_store(vector_store)

        chunks = [make_chunk(chunk_id="chunk1")]

        mock_bulk.return_value = (1, [])

        await adapter.insert_chunks(
            InsertChunksRequest(
                vector_store_id="test_store",
                chunks=chunks,
            )
        )

        mock_bulk.assert_called_once()
        self.assertEqual(mock_bulk.call_args[0][0], mock_client)
        actions = mock_bulk.call_args[0][1]
        self.assertEqual(len(actions), 1)
        self.assertEqual(actions[0]["_index"], "test_store")
        self.assertEqual(actions[0]["_id"], "chunk1")

    @patch("ogx.providers.remote.vector_io.opensearch.opensearch.OpenSearch")
    async def test_query_chunks_delegation(self, mock_opensearch):
        """Test that query_chunks delegates to VectorStoreWithIndex."""
        adapter = OpenSearchVectorIOAdapter(self.config, self.inference_api, self.files_api, file_processor_api=None)
        mock_client = MagicMock()
        mock_opensearch.return_value = mock_client
        mock_client.indices.exists.return_value = True
        adapter.kvstore = AsyncMock()
        with patch.object(adapter, "initialize_openai_vector_stores", new_callable=AsyncMock):
            await adapter.initialize()

        vector_store = VectorStore(
            identifier="test_store",
            provider_id="test_provider",
            provider_resource_id="test_resource",
            embedding_dimension=4,
            embedding_model="test_model",
        )
        await adapter.register_vector_store(vector_store)

        # Mock the VectorStoreWithIndex.query_chunks method directly
        mock_store = adapter.cache["test_store"]
        mock_store.query_chunks = AsyncMock(
            return_value=QueryChunksResponse(
                chunks=[
                    make_chunk(chunk_id="chunk1"),
                ],
                scores=[0.9],
            )
        )

        response = await adapter.query_chunks(
            QueryChunksRequest(
                vector_store_id="test_store",
                query="test query",
                params={
                    "mode": "vector",
                    "score_threshold": 0.5,
                },
            )
        )

        # Verify the store's query_chunks was called with the request
        mock_store.query_chunks.assert_called_once()
        self.assertEqual(len(response.chunks), 1)
        self.assertEqual(response.chunks[0].chunk_id, "chunk1")
        self.assertEqual(response.scores[0], 0.9)

    @patch("ogx.providers.remote.vector_io.opensearch.opensearch.OpenSearch")
    async def test_vector_store_identifier_lowercase(self, mock_opensearch):
        """Verify that vector store identifiers are normalized to lowercase for OpenSearch."""
        adapter = OpenSearchVectorIOAdapter(self.config, self.inference_api, self.files_api, file_processor_api=None)
        mock_client = MagicMock()
        mock_opensearch.return_value = mock_client
        adapter.kvstore = AsyncMock()
        with patch.object(adapter, "initialize_openai_vector_stores", new_callable=AsyncMock):
            await adapter.initialize()

        vector_store = VectorStore(
            identifier="My_UPPER_Case_Store",
            provider_id="test_provider",
            provider_resource_id="test_resource",
            embedding_dimension=4,
            embedding_model="test_model",
        )

        mock_client.indices.exists.return_value = False
        await adapter.register_vector_store(vector_store)

        # OpenSearch indices must be lowercase
        call_args = mock_client.indices.create.call_args[1]
        self.assertEqual(call_args["index"], "my_upper_case_store")


if __name__ == "__main__":
    unittest.main()
