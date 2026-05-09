# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
import json
from typing import Any

import weaviate
import weaviate.classes as wvc
from numpy.typing import NDArray
from weaviate.classes.init import Auth
from weaviate.classes.query import Filter, HybridFusion

from ogx.core.storage.kvstore import kvstore_impl
from ogx.log import get_logger
from ogx.providers.utils.memory.openai_vector_store_mixin import OpenAIVectorStoreMixin
from ogx.providers.utils.memory.vector_store import (
    RERANKER_TYPE_RRF,
    ChunkForDeletion,
    EmbeddingIndex,
    VectorStoreWithIndex,
)
from ogx.providers.utils.vector_io import load_embedded_chunk_with_backward_compat
from ogx.providers.utils.vector_io.vector_utils import sanitize_collection_name
from ogx_api import (
    ComparisonFilter,
    CompoundFilter,
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
from ogx_api.internal.kvstore import KVStore

from .config import WeaviateVectorIOConfig

log = get_logger(name=__name__, category="vector_io::weaviate")

VERSION = "v3"
VECTOR_DBS_PREFIX = f"vector_stores:weaviate:{VERSION}::"
VECTOR_INDEX_PREFIX = f"vector_index:weaviate:{VERSION}::"
OPENAI_VECTOR_STORES_PREFIX = f"openai_vector_stores:weaviate:{VERSION}::"
OPENAI_VECTOR_STORES_FILES_PREFIX = f"openai_vector_stores_files:weaviate:{VERSION}::"
OPENAI_VECTOR_STORES_FILES_CONTENTS_PREFIX = f"openai_vector_stores_files_contents:weaviate:{VERSION}::"


class WeaviateIndex(EmbeddingIndex):
    """Embedding index backed by a Weaviate collection."""

    def __init__(self, client: weaviate.WeaviateClient, collection_name: str, kvstore: KVStore | None = None):
        self.client = client
        self.collection_name = sanitize_collection_name(collection_name, weaviate_format=True)
        self.kvstore = kvstore

    async def initialize(self):
        pass

    async def add_chunks(self, chunks: list[EmbeddedChunk]):
        if not chunks:
            return

        data_objects = []
        for chunk in chunks:
            data_objects.append(
                wvc.data.DataObject(
                    properties={
                        "chunk_id": chunk.chunk_id,
                        "chunk_content": chunk.model_dump_json(),
                    },
                    vector=chunk.embedding,  # Already a list[float]
                )
            )

        # Inserting chunks into a prespecified Weaviate collection
        collection = self.client.collections.get(self.collection_name)

        # TODO: make this async friendly
        collection.data.insert_many(data_objects)

    async def delete_chunks(self, chunks_for_deletion: list[ChunkForDeletion]) -> None:
        sanitized_collection_name = sanitize_collection_name(self.collection_name, weaviate_format=True)
        collection = self.client.collections.get(sanitized_collection_name)
        chunk_ids = [chunk.chunk_id for chunk in chunks_for_deletion]
        collection.data.delete_many(where=Filter.by_property("chunk_id").contains_any(chunk_ids))

    async def query_vector(
        self, embedding: NDArray, k: int, score_threshold: float, filters: Any = None
    ) -> QueryChunksResponse:
        """
        Performs vector search using Weaviate's built-in vector search.
        Args:
            embedding: The query embedding vector
            k: Limit of number of results to return
            score_threshold: Minimum similarity score threshold
            filters: Optional filters (not yet supported for Weaviate provider)
        Returns:
            QueryChunksResponse with chunks and scores.
        """
        # Filters are not yet implemented for Weaviate provider
        if filters is not None:
            raise NotImplementedError("Weaviate provider does not yet support native filtering")

        log.debug(
            f"WEAVIATE VECTOR SEARCH CALLED: embedding_shape={embedding.shape}, k={k}, threshold={score_threshold}"
        )
        sanitized_collection_name = sanitize_collection_name(self.collection_name, weaviate_format=True)
        collection = self.client.collections.get(sanitized_collection_name)

        try:
            results = collection.query.near_vector(
                near_vector=embedding.tolist(), limit=k, return_metadata=wvc.query.MetadataQuery(distance=True)
            )
        except Exception as e:
            log.error(f"Weaviate client vector search failed: {e}")
            raise

        chunks = []
        scores = []
        for doc in results.objects:
            chunk_json = doc.properties["chunk_content"]
            try:
                chunk_dict = json.loads(chunk_json)
                chunk = load_embedded_chunk_with_backward_compat(chunk_dict)
            except Exception:
                log.exception(f"Failed to parse document: {chunk_json}")
                continue

            if doc.metadata.distance is None:
                continue
            # Convert cosine distance ∈ [0,2] -> normalized cosine similarity ∈ [0,1]
            score = 1.0 - (float(doc.metadata.distance) / 2.0)
            if score < score_threshold:
                continue

            chunks.append(chunk)
            scores.append(score)

        log.debug(f"WEAVIATE VECTOR SEARCH RESULTS: Found {len(chunks)} chunks with scores {scores}")
        return QueryChunksResponse(chunks=chunks, scores=scores)

    async def delete(self, chunk_ids: list[str] | None = None) -> None:
        """
        Delete chunks by IDs if provided, otherwise drop the entire collection.
        """
        sanitized_collection_name = sanitize_collection_name(self.collection_name, weaviate_format=True)
        if chunk_ids is None:
            # Drop entire collection if it exists
            if self.client.collections.exists(sanitized_collection_name):
                self.client.collections.delete(sanitized_collection_name)
            return
        collection = self.client.collections.get(sanitized_collection_name)
        collection.data.delete_many(where=Filter.by_property("id").contains_any(chunk_ids))

    async def query_keyword(
        self,
        query_string: str,
        k: int,
        score_threshold: float,
        filters: ComparisonFilter | CompoundFilter | None = None,
    ) -> QueryChunksResponse:
        """
        Performs BM25-based keyword search using Weaviate's built-in full-text search.
        Args:
            query_string: The text query for keyword search
            k: Limit of number of results to return
            score_threshold: Minimum similarity score threshold
        Returns:
            QueryChunksResponse with chunks and scores
        """
        # Weaviate provider does not yet support native filtering
        if filters is not None:
            raise NotImplementedError("Weaviate provider does not yet support native filtering")

        log.debug(f"WEAVIATE KEYWORD SEARCH CALLED: query='{query_string}', k={k}, threshold={score_threshold}")
        sanitized_collection_name = sanitize_collection_name(self.collection_name, weaviate_format=True)
        collection = self.client.collections.get(sanitized_collection_name)

        # Perform BM25 keyword search on chunk_content field
        try:
            results = collection.query.bm25(
                query=query_string, limit=k, return_metadata=wvc.query.MetadataQuery(score=True)
            )
        except Exception as e:
            log.error(f"Weaviate client keyword search failed: {e}")
            raise

        chunks = []
        scores = []
        for doc in results.objects:
            chunk_json = doc.properties["chunk_content"]
            try:
                chunk_dict = json.loads(chunk_json)
                chunk = load_embedded_chunk_with_backward_compat(chunk_dict)
            except Exception:
                log.exception(f"Failed to parse document: {chunk_json}")
                continue

            score = doc.metadata.score if doc.metadata.score is not None else 0.0
            if score < score_threshold:
                continue

            chunks.append(chunk)
            scores.append(score)

        log.debug(f"WEAVIATE KEYWORD SEARCH RESULTS: Found {len(chunks)} chunks with scores {scores}.")
        return QueryChunksResponse(chunks=chunks, scores=scores)

    async def query_hybrid(
        self,
        embedding: NDArray,
        query_string: str,
        k: int,
        score_threshold: float,
        reranker_type: str,
        reranker_params: dict[str, Any] | None = None,
        filters: ComparisonFilter | CompoundFilter | None = None,
    ) -> QueryChunksResponse:
        """
        Hybrid search combining vector similarity and keyword search using Weaviate's native hybrid search.
        Args:
            embedding: The query embedding vector
            query_string: The text query for keyword search
            k: Limit of number of results to return
            score_threshold: Minimum similarity score threshold
            reranker_type: Type of reranker to use ("rrf" or "normalized")
            reranker_params: Parameters for the reranker
        Returns:
            QueryChunksResponse with combined results
        """
        # Weaviate provider does not yet support native filtering
        if filters is not None:
            raise NotImplementedError("Weaviate provider does not yet support native filtering")

        log.debug(
            f"WEAVIATE HYBRID SEARCH CALLED: query='{query_string}', embedding_shape={embedding.shape}, k={k}, threshold={score_threshold}, reranker={reranker_type}"
        )
        sanitized_collection_name = sanitize_collection_name(self.collection_name, weaviate_format=True)
        collection = self.client.collections.get(sanitized_collection_name)

        # Ranked (RRF) reranker fusion type
        if reranker_type == RERANKER_TYPE_RRF:
            rerank = HybridFusion.RANKED
        # Relative score (Normalized) reranker fusion type
        else:
            rerank = HybridFusion.RELATIVE_SCORE

        # Perform hybrid search using Weaviate's native hybrid search
        try:
            results = collection.query.hybrid(
                query=query_string,
                alpha=0.5,  # Range <0, 1>, where 0.5 will equally favor vector and keyword search
                vector=embedding.tolist(),
                limit=k,
                fusion_type=rerank,
                return_metadata=wvc.query.MetadataQuery(score=True),
            )
        except Exception as e:
            log.error(f"Weaviate client hybrid search failed: {e}")
            raise

        chunks = []
        scores = []
        for doc in results.objects:
            chunk_json = doc.properties["chunk_content"]
            try:
                chunk_dict = json.loads(chunk_json)
                chunk = load_embedded_chunk_with_backward_compat(chunk_dict)
            except Exception:
                log.exception(f"Failed to parse document: {chunk_json}")
                continue

            score = doc.metadata.score if doc.metadata.score is not None else 0.0
            if score < score_threshold:
                continue

            chunks.append(chunk)
            scores.append(score)

        log.debug(f"WEAVIATE HYBRID SEARCH RESULTS: Found {len(chunks)} chunks with scores {scores}")
        return QueryChunksResponse(chunks=chunks, scores=scores)


class WeaviateVectorIOAdapter(OpenAIVectorStoreMixin, VectorIO, VectorStoresProtocolPrivate):
    """VectorIO adapter that uses Weaviate for similarity search and vector storage."""

    def __init__(
        self,
        config: WeaviateVectorIOConfig,
        inference_api: Inference,
        files_api: Files | None,
        file_processor_api: FileProcessors | None = None,
    ) -> None:
        super().__init__(
            inference_api=inference_api, files_api=files_api, kvstore=None, file_processor_api=file_processor_api
        )
        self.config = config
        self.client_cache = {}
        self.cache = {}
        self.vector_store_table = None
        self.metadata_collection_name = "openai_vector_stores_metadata"

    def _get_client(self) -> weaviate.WeaviateClient:
        if "localhost" in self.config.weaviate_cluster_url:
            log.info("Using Weaviate locally in container")
            host, port = self.config.weaviate_cluster_url.split(":")
            key = "local_test"
            client = weaviate.connect_to_local(host=host, port=port)
        else:
            log.info("Using Weaviate remote cluster with URL")
            key = f"{self.config.weaviate_cluster_url}::{self.config.weaviate_api_key}"
            if key in self.client_cache:
                return self.client_cache[key]
            client = weaviate.connect_to_weaviate_cloud(
                cluster_url=self.config.weaviate_cluster_url,
                auth_credentials=Auth.api_key(self.config.weaviate_api_key),
            )
        self.client_cache[key] = client
        return client

    async def initialize(self) -> None:
        """Set up KV store and load existing vector DBs and OpenAI vector stores."""
        # Initialize KV store for metadata if configured
        if self.config.persistence is not None:
            self.kvstore = await kvstore_impl(self.config.persistence)
        else:
            self.kvstore = None
            log.info("No kvstore configured, registry will not persist across restarts")

        # Load existing vector DB definitions
        if self.kvstore is not None:
            start_key = VECTOR_DBS_PREFIX
            end_key = f"{VECTOR_DBS_PREFIX}\xff"
            stored = await self.kvstore.values_in_range(start_key, end_key)
            for raw in stored:
                vector_store = VectorStore.model_validate_json(raw)
                client = self._get_client()
                idx = WeaviateIndex(client=client, collection_name=vector_store.identifier, kvstore=self.kvstore)
                self.cache[vector_store.identifier] = VectorStoreWithIndex(
                    vector_store=vector_store, index=idx, inference_api=self.inference_api
                )

            # Load OpenAI vector stores metadata into cache
            await self.initialize_openai_vector_stores()

    async def shutdown(self) -> None:
        for client in self.client_cache.values():
            client.close()
        # Clean up mixin resources (file batch tasks)
        await super().shutdown()

    async def register_vector_store(self, vector_store: VectorStore) -> None:
        client = self._get_client()
        sanitized_collection_name = sanitize_collection_name(vector_store.identifier, weaviate_format=True)
        # Create collection if it doesn't exist
        if not client.collections.exists(sanitized_collection_name):
            client.collections.create(
                name=sanitized_collection_name,
                vectorizer_config=wvc.config.Configure.Vectorizer.none(),
                properties=[
                    wvc.config.Property(name="chunk_content", data_type=wvc.config.DataType.TEXT),
                ],
            )

        self.cache[vector_store.identifier] = VectorStoreWithIndex(
            vector_store, WeaviateIndex(client=client, collection_name=sanitized_collection_name), self.inference_api
        )

    async def unregister_vector_store(self, vector_store_id: str) -> None:
        client = self._get_client()
        sanitized_collection_name = sanitize_collection_name(vector_store_id, weaviate_format=True)
        if vector_store_id not in self.cache or client.collections.exists(sanitized_collection_name) is False:
            return
        client.collections.delete(sanitized_collection_name)
        await self.cache[vector_store_id].index.delete()
        del self.cache[vector_store_id]

    async def _get_and_cache_vector_store_index(self, vector_store_id: str) -> VectorStoreWithIndex | None:
        if vector_store_id in self.cache:
            return self.cache[vector_store_id]

        # Try to load from kvstore
        if self.kvstore is None:
            raise RuntimeError("KVStore not initialized. Call initialize() before using vector stores.")

        key = f"{VECTOR_DBS_PREFIX}{vector_store_id}"
        vector_store_data = await self.kvstore.get(key)
        if not vector_store_data:
            raise VectorStoreNotFoundError(vector_store_id)

        vector_store = VectorStore.model_validate_json(vector_store_data)
        client = self._get_client()
        sanitized_collection_name = sanitize_collection_name(vector_store.identifier, weaviate_format=True)
        if not client.collections.exists(sanitized_collection_name):
            raise ValueError(f"Collection with name `{sanitized_collection_name}` not found")

        index = VectorStoreWithIndex(
            vector_store=vector_store,
            index=WeaviateIndex(client=client, collection_name=vector_store.identifier),
            inference_api=self.inference_api,
        )
        self.cache[vector_store_id] = index
        return index

    async def insert_chunks(self, request: InsertChunksRequest) -> None:
        index = await self._get_and_cache_vector_store_index(request.vector_store_id)
        if not index:
            raise VectorStoreNotFoundError(request.vector_store_id)

        await index.insert_chunks(request)

    async def query_chunks(self, request: QueryChunksRequest) -> QueryChunksResponse:
        index = await self._get_and_cache_vector_store_index(request.vector_store_id)
        if not index:
            raise VectorStoreNotFoundError(request.vector_store_id)

        return await index.query_chunks(request)

    async def delete_chunks(self, request: DeleteChunksRequest) -> None:
        index = await self._get_and_cache_vector_store_index(request.vector_store_id)
        if not index:
            raise ValueError(f"Vector DB {request.vector_store_id} not found")

        await index.index.delete_chunks(request.chunks)
