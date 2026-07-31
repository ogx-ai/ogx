# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import asyncio
from typing import TYPE_CHECKING, Any

from numpy.typing import NDArray

from ogx.core.storage.kvstore import kvstore_impl
from ogx.log import get_logger
from ogx.providers.utils.memory.openai_vector_store_mixin import OpenAIVectorStoreMixin
from ogx.providers.utils.memory.vector_store import ChunkForDeletion, EmbeddingIndex, VectorStoreWithIndex
from ogx.providers.utils.vector_io.filters import Filter
from ogx_api import (
    DeleteChunksRequest,
    EmbeddedChunk,
    FileProcessors,
    Files,
    Inference,
    InsertChunksRequest,
    QueryChunksRequest,
    QueryChunksResponse,
    VectorIO,
    VectorStore,
    VectorStoreNotFoundError,
    VectorStoresProtocolPrivate,
)

from .config import OpenSearchVectorIOConfig

if TYPE_CHECKING:
    from opensearchpy import OpenSearch, helpers

try:
    from opensearchpy import OpenSearch, helpers  # noqa: F401
except ImportError:
    OpenSearch = None
    helpers = None

logger = get_logger(name=__name__, category="vector_io::opensearch")

# KV store prefixes for vector databases
VERSION = "v3"
VECTOR_DBS_PREFIX = f"vector_stores:opensearch:{VERSION}::"
VECTOR_INDEX_PREFIX = f"vector_index:opensearch:{VERSION}::"
OPENAI_VECTOR_STORES_PREFIX = f"openai_vector_stores:opensearch:{VERSION}::"
OPENAI_VECTOR_STORES_FILES_PREFIX = f"openai_vector_stores_files:opensearch:{VERSION}::"
OPENAI_VECTOR_STORES_FILES_CONTENTS_PREFIX = f"openai_vector_stores_files_contents:opensearch:{VERSION}::"


class OpenSearchIndex(EmbeddingIndex):
    """Embedding index backed by an OpenSearch index with k-NN support."""

    def __init__(self, client: Any, vector_store: VectorStore):
        self.client = client
        self.collection_name = vector_store.identifier.lower()  # OpenSearch indices must be lowercase
        self.dimension = vector_store.embedding_dimension

    async def initialize(self) -> None:
        """Create the OpenSearch index with k-NN mapping if it doesn't exist."""
        exists = await asyncio.to_thread(self.client.indices.exists, index=self.collection_name)
        if not exists:
            mapping = {
                "settings": {"index": {"knn": True}},
                "mappings": {
                    "properties": {
                        "embedding": {
                            "type": "knn_vector",
                            "dimension": self.dimension,
                            "method": {
                                "name": "hnsw",
                                "engine": "lucene",
                                "space_type": "l2",
                            },
                        },
                        "chunk_id": {"type": "keyword"},
                        "content": {"type": "text"},
                        "metadata": {"type": "object"},
                        "chunk_metadata": {"type": "object"},
                        "embedding_dimension": {"type": "integer"},
                        "embedding_model": {"type": "keyword"},
                    }
                },
            }
            try:
                await asyncio.to_thread(self.client.indices.create, index=self.collection_name, body=mapping)
                logger.info("Created OpenSearch index", collection_name=self.collection_name)
            except Exception as e:
                logger.warning("Index creation failed (might already exist)", error=str(e))

    async def delete(self) -> None:
        """Delete the entire OpenSearch index."""
        try:
            await asyncio.to_thread(self.client.indices.delete, index=self.collection_name, ignore_unavailable=True)
        except Exception as e:
            logger.error("Failed to delete index", collection_name=self.collection_name, error=str(e))

    async def add_chunks(self, chunks: list[EmbeddedChunk]) -> None:
        """Add chunks to the OpenSearch index."""
        if not chunks:
            return

        actions = []
        for chunk in chunks:
            doc = {
                "_index": self.collection_name,
                "_id": chunk.chunk_id,
                "_source": chunk.model_dump(
                    exclude_none=True,
                    include={
                        "content",
                        "chunk_id",
                        "metadata",
                        "chunk_metadata",
                        "embedding",
                        "embedding_dimension",
                        "embedding_model",
                    },
                ),
            }
            actions.append(doc)

        success, failed = await asyncio.to_thread(helpers.bulk, self.client, actions, refresh=True)
        if failed:
            logger.error("Failed to index documents", failed_count=len(failed), index_name=self.collection_name)

    async def query_vector(
        self,
        embedding: "NDArray[Any]",
        k: int,
        score_threshold: float,
        filters: Filter | None = None,
    ) -> QueryChunksResponse:
        if filters is not None:
            raise NotImplementedError("OpenSearch provider does not yet support native filtering")

        query = {
            "size": k,
            "query": {
                "knn": {
                    "embedding": {
                        "vector": embedding.tolist(),
                        "k": k,
                    }
                }
            },
            "min_score": score_threshold,
        }

        response = await asyncio.to_thread(self.client.search, index=self.collection_name, body=query)

        return await self._results_to_chunks(response)

    async def query_keyword(
        self,
        query_string: str,
        k: int,
        score_threshold: float,
        filters: Filter | None = None,
    ) -> QueryChunksResponse:
        if filters is not None:
            raise NotImplementedError("OpenSearch provider does not yet support native filtering")

        query = {
            "size": k,
            "query": {"match": {"content": query_string}},
            "min_score": score_threshold,
        }

        response = await asyncio.to_thread(self.client.search, index=self.collection_name, body=query)

        return await self._results_to_chunks(response)

    async def query_hybrid(
        self,
        embedding: "NDArray[Any]",
        query_string: str,
        k: int,
        score_threshold: float,
        reranker_type: str,
        reranker_params: dict[str, Any] | None = None,
        filters: Filter | None = None,
    ) -> QueryChunksResponse:
        if filters is not None:
            raise NotImplementedError("OpenSearch provider does not yet support native filtering")

        # Simple hybrid using compound bool query
        query = {
            "size": k,
            "query": {
                "bool": {
                    "should": [
                        {
                            "knn": {
                                "embedding": {
                                    "vector": embedding.tolist(),
                                    "k": k,
                                    "boost": 0.5,
                                }
                            }
                        },
                        {
                            "match": {
                                "content": {
                                    "query": query_string,
                                    "boost": 0.5,
                                }
                            }
                        },
                    ]
                }
            },
            "min_score": score_threshold,
        }

        response = await asyncio.to_thread(self.client.search, index=self.collection_name, body=query)

        return await self._results_to_chunks(response)

    async def delete_chunks(self, chunks_for_deletion: list[ChunkForDeletion]) -> None:
        """Delete chunks from the OpenSearch index."""
        actions = []
        for chunk in chunks_for_deletion:
            actions.append(
                {
                    "_op_type": "delete",
                    "_index": self.collection_name,
                    "_id": chunk.chunk_id,
                }
            )

        await asyncio.to_thread(helpers.bulk, self.client, actions, raise_on_error=False, refresh=True)

    async def _results_to_chunks(self, results: dict) -> QueryChunksResponse:
        """Convert search results to QueryChunksResponse."""
        chunks, scores = [], []
        for hit in results.get("hits", {}).get("hits", []):
            source = hit.get("_source", {})
            try:
                chunk = EmbeddedChunk(
                    content=source.get("content"),
                    chunk_id=hit.get("_id"),
                    embedding=source.get("embedding", []),
                    embedding_dimension=source.get("embedding_dimension", len(source.get("embedding", []))),
                    embedding_model=source.get("embedding_model", "unknown"),
                    chunk_metadata=source.get("chunk_metadata", {}),
                    metadata=source.get("metadata", {}),
                )
            except Exception:
                logger.exception("Failed to parse chunk")
                continue

            chunks.append(chunk)
            scores.append(hit.get("_score"))

        return QueryChunksResponse(chunks=chunks, scores=scores)


class OpenSearchVectorIOAdapter(OpenAIVectorStoreMixin, VectorIO, VectorStoresProtocolPrivate):
    """Vector I/O adapter for remote OpenSearch instances."""

    def __init__(
        self,
        config: OpenSearchVectorIOConfig,
        inference_api: Inference,
        files_api: Files | None = None,
        file_processor_api: FileProcessors | None = None,
        policy: list | None = None,
    ) -> None:
        super().__init__(
            inference_api=inference_api,
            files_api=files_api,
            kvstore=None,
            file_processor_api=file_processor_api,
        )
        self.config = config
        self.client: OpenSearch | None = None
        self.cache: dict[str, VectorStoreWithIndex] = {}
        self.vector_store_table = None
        self.metadata_collection_name = "openai_vector_stores_metadata"
        self._policy = policy or []

    async def initialize(self) -> None:
        """Initialize the OpenSearch client and load any stored vector stores."""
        if OpenSearch is None:
            raise ImportError("opensearch-py is not installed. Please install it with `pip install opensearch-py`.")

        auth = None
        if self.config.username and self.config.password:
            auth = (self.config.username, self.config.password)

        self.client = OpenSearch(
            hosts=[{"host": self.config.host, "port": self.config.port}],
            http_compress=True,
            http_auth=auth,
            use_ssl=self.config.use_ssl,
            verify_certs=self.config.verify_certs,
        )

        if self.config.persistence is not None:
            self.kvstore = await kvstore_impl(self.config.persistence)

        if self.config.metadata_store:
            from ogx.core.storage.sqlstore import authorized_sqlstore

            self.metadata_store = await authorized_sqlstore(self.config.metadata_store, self._policy)

        # Try to ping or get info to verify connection
        try:
            await asyncio.to_thread(self.client.info)
        except Exception as e:
            logger.warning("Could not connect to OpenSearch at startup", error=str(e))

        # Load any stored vector stores from kvstore
        start_key = VECTOR_DBS_PREFIX
        end_key = f"{VECTOR_DBS_PREFIX}\xff"
        if self.kvstore:
            stored_vector_stores = await self.kvstore.values_in_range(start_key, end_key)
            for vector_store_data in stored_vector_stores:
                try:
                    from ogx_api import VectorStore

                    vector_store = VectorStore.model_validate_json(vector_store_data)
                    os_index = OpenSearchIndex(self.client, vector_store)
                    await os_index.initialize()
                    index = VectorStoreWithIndex(
                        vector_store=vector_store,
                        index=os_index,
                        inference_api=self.inference_api,
                    )
                    self.cache[vector_store.identifier] = index
                except Exception as e:
                    logger.error("Failed to load stored vector store", vector_store=vector_store_data, error=str(e))

        await self.initialize_openai_vector_stores()

    async def health(self) -> dict[str, Any]:
        """Return health status of the OpenSearch connection."""
        try:
            if self.client is None:
                return {"status": "error", "message": "Not initialized"}
            await asyncio.to_thread(self.client.ping)
            return {"status": "ok"}
        except Exception as e:
            return {"status": "error", "message": f"Health check failed: {str(e)}"}

    async def shutdown(self) -> None:
        """Close the OpenSearch client."""
        if self.client:
            self.client.close()
        await super().shutdown()

    async def register_vector_store(self, vector_store: VectorStore) -> None:
        """Register a new vector store."""
        assert self.kvstore is not None
        key = f"{VECTOR_DBS_PREFIX}{vector_store.identifier}"
        await self.kvstore.set(key=key, value=vector_store.model_dump_json())

        os_index = OpenSearchIndex(self.client, vector_store)
        await os_index.initialize()

        index = VectorStoreWithIndex(
            vector_store=vector_store,
            index=os_index,
            inference_api=self.inference_api,
        )
        self.cache[vector_store.identifier] = index

    async def unregister_vector_store(self, vector_store_id: str) -> None:
        """Unregister and delete a vector store."""
        if vector_store_id in self.cache:
            await self.cache[vector_store_id].index.delete()
            del self.cache[vector_store_id]

        assert self.kvstore is not None
        await self.kvstore.delete(f"{VECTOR_DBS_PREFIX}{vector_store_id}")

    async def list_vector_stores(self) -> list[VectorStore]:
        """List all registered vector stores."""
        return [item.vector_store for item in self.cache.values()]

    async def insert_chunks(self, request: InsertChunksRequest) -> None:
        """Insert chunks into a vector store."""
        store = await self._get_and_cache_vector_store_index(request.vector_store_id)
        if not store:
            raise VectorStoreNotFoundError(request.vector_store_id)

        await store.index.add_chunks(request.chunks)

    async def query_chunks(self, request: QueryChunksRequest) -> QueryChunksResponse:
        """Query chunks from a vector store."""
        store = await self._get_and_cache_vector_store_index(request.vector_store_id)
        if not store:
            raise VectorStoreNotFoundError(request.vector_store_id)

        return await store.query_chunks(request)

    async def delete_chunks(self, request: DeleteChunksRequest) -> None:
        """Delete chunks from a vector store."""
        index = await self._get_and_cache_vector_store_index(request.vector_store_id)
        if not index:
            raise ValueError(f"Vector DB {request.vector_store_id} not found")

        await index.index.delete_chunks(request.chunks)

    async def _get_and_cache_vector_store_index(self, vector_store_id: str) -> VectorStoreWithIndex | None:
        """Get a cached vector store index, or load it if not in cache."""
        if vector_store_id in self.cache:
            return self.cache[vector_store_id]

        if self.kvstore is None:
            raise RuntimeError("KVStore not initialized. Call initialize() before using vector stores.")

        key = f"{VECTOR_DBS_PREFIX}{vector_store_id}"
        vector_store_data = await self.kvstore.get(key)
        if not vector_store_data:
            raise VectorStoreNotFoundError(vector_store_id)

        from ogx_api import VectorStore

        vector_store = VectorStore.model_validate_json(vector_store_data)
        os_index = OpenSearchIndex(client=self.client, vector_store=vector_store)
        await os_index.initialize()
        index = VectorStoreWithIndex(
            vector_store=vector_store,
            index=os_index,
            inference_api=self.inference_api,
        )
        self.cache[vector_store_id] = index
        return index
