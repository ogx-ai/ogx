# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import heapq
import json
from typing import Any

import httpx
from numpy.typing import NDArray

from ogx.core.storage.kvstore import kvstore_impl
from ogx.log import get_logger
from ogx.providers.utils.memory.openai_vector_store_mixin import OpenAIVectorStoreMixin
from ogx.providers.utils.memory.vector_store import ChunkForDeletion, EmbeddingIndex, VectorStoreWithIndex
from ogx.providers.utils.vector_io import load_embedded_chunk_with_backward_compat
from ogx.providers.utils.vector_io.filters import Filter
from ogx.providers.utils.vector_io.vector_utils import WeightedInMemoryAggregator
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
    VectorStoresProtocolPrivate,
)
from ogx_api.internal.kvstore import KVStore

from .config import InfinispanVectorIOConfig
from .schemas import load_schema

log = get_logger(name=__name__, category="vector_io::infinispan")

VERSION = "v3"
VECTOR_DBS_PREFIX = f"vector_stores:infinispan:{VERSION}::"
VECTOR_INDEX_PREFIX = f"vector_index:infinispan:{VERSION}::"
OPENAI_VECTOR_STORES_PREFIX = f"openai_vector_stores:infinispan:{VERSION}::"
OPENAI_VECTOR_STORES_FILES_PREFIX = f"openai_vector_stores_files:infinispan:{VERSION}::"
OPENAI_VECTOR_STORES_FILES_CONTENTS_PREFIX = f"openai_vector_stores_files_contents:infinispan:{VERSION}::"


class InfinispanIndex(EmbeddingIndex):
    """
    Infinispan-specific implementation of EmbeddingIndex.
    Uses HTTP REST API for all operations.
    """

    def __init__(
        self,
        client: httpx.AsyncClient,
        cache_name: str,
        base_url: str,
        embedding_dimension: int,
        kvstore: KVStore | None = None,
    ):
        self.client = client
        self.cache_name = cache_name
        self.base_url = base_url.rstrip("/")
        self.embedding_dimension = embedding_dimension
        self.kvstore = kvstore

    async def initialize(self):
        """
        Check if the Infinispan cache exists, create if needed.

        Uses Infinispan REST API v3:
        1. HEAD /rest/v3/caches/{cacheName} to check cache existence
        2. POST /rest/v3/caches/{cacheName} to create cache with configuration if not exists

        Cache configuration supports:
        - JSON encoding for storing chunk data and embeddings
        - Indexing enabled for full-text search
        """
        # Check if cache exists using HEAD request
        response = await self.client.head(f"{self.base_url}/rest/v3/caches/{self.cache_name}")

        if response.status_code == 204:
            # Cache already exists
            log.info(f"Cache '{self.cache_name}' already exists")
            return

        if response.status_code != 404:
            # Unexpected error
            log.error(f"Failed to check cache existence: {response.status_code} - {response.text}")
            response.raise_for_status()

        # Cache doesn't exist, register schema first then create cache
        log.info(f"Creating cache '{self.cache_name}'")

        # Register Protobuf schema first (cache will reference this)
        await self._register_protobuf_schema()

        # Load cache configuration template and replace placeholders
        cache_config_template = load_schema("cache_config.xml")
        entity_name = f"VectorItem{self.embedding_dimension}"
        cache_config_xml = cache_config_template.replace("{CACHE_NAME}", self.cache_name).replace(
            "{ENTITY_NAME}", entity_name
        )

        # Create cache with XML configuration
        create_response = await self.client.post(
            f"{self.base_url}/rest/v3/caches/{self.cache_name}",
            content=cache_config_xml,
            headers={"Content-Type": "application/xml"},
        )

        if create_response.status_code not in [200, 204]:
            log.error(f"Failed to create cache: {create_response.status_code} - {create_response.text}")
            create_response.raise_for_status()

        log.info(f"Cache '{self.cache_name}' created successfully")

    async def _register_protobuf_schema(self):
        """Register the Protobuf schema with Infinispan for vector indexing."""
        # Load the schema template and replace dimension placeholder
        schema_template = load_schema("vector_chunk.proto")
        schema_content = schema_template.replace("{DIMENSION}", str(self.embedding_dimension))

        # Schema name must match the message name
        schema_name = f"vector_chunk_{self.embedding_dimension}.proto"

        # Register schema with Infinispan's protobuf metadata cache
        schema_response = await self.client.put(
            f"{self.base_url}/rest/v3/caches/___protobuf_metadata/entries/{schema_name}",
            content=schema_content,
            headers={"Content-Type": "text/plain"},
        )

        if schema_response.status_code not in [200, 204]:
            log.error(f"Failed to register Protobuf schema: {schema_response.status_code} - {schema_response.text}")
            schema_response.raise_for_status()

        log.info(f"Protobuf schema '{schema_name}' registered successfully")

    async def add_chunks(self, chunks: list[EmbeddedChunk]):
        """
        Insert embedded chunks into Infinispan cache.

        Uses Infinispan REST API v3:
        - PUT /rest/v3/caches/{cacheName}/entries/{key} for each chunk
        - Maps EmbeddedChunk to VectorItem protobuf schema

        Args:
            chunks: List of EmbeddedChunk objects to insert
        """
        if not chunks:
            return

        log.info(f"Inserting {len(chunks)} chunks into cache '{self.cache_name}'")

        for chunk in chunks:
            # Generate key from chunk_id
            key = chunk.chunk_id

            # Map EmbeddedChunk to VectorItem protobuf schema
            # VectorItem fields: id, floatVector, text, metadata, chunkMetadata, embeddingModel
            vector_item = {
                "_type": f"VectorItem{self.embedding_dimension}",
                "id": chunk.chunk_id,
                "floatVector": chunk.embedding.tolist()
                if hasattr(chunk.embedding, "tolist")
                else list(chunk.embedding),
                "text": chunk.content if hasattr(chunk, "content") else "",
                "metadata": json.dumps(chunk.metadata) if chunk.metadata else "{}",
                "chunkMetadata": json.dumps(chunk.chunk_metadata.model_dump()) if chunk.chunk_metadata else "{}",
                "embeddingModel": chunk.embedding_model if hasattr(chunk, "embedding_model") else "unknown",
            }

            # Insert into Infinispan cache
            log.debug(f"PUT request to insert chunk {key}: {vector_item}")
            response = await self.client.put(
                f"{self.base_url}/rest/v3/caches/{self.cache_name}/entries/{key}",
                json=vector_item,
                headers={"Content-Type": "application/json"},
            )

            if response.status_code not in [200, 204]:
                log.error(f"Failed to insert chunk {key}: {response.status_code} - {response.text}")
                response.raise_for_status()

        log.info(f"Successfully inserted {len(chunks)} chunks")

    async def delete_chunks(self, chunks_for_deletion: list[ChunkForDeletion]) -> None:
        """
        Delete specific chunks from Infinispan cache.

        Uses Infinispan REST API v3:
        - DELETE /rest/v3/caches/{cacheName}/entries/{key} for each chunk

        Args:
            chunks_for_deletion: List of ChunkForDeletion objects specifying which chunks to remove
        """
        if not chunks_for_deletion:
            return

        log.info(f"Deleting {len(chunks_for_deletion)} chunks from cache '{self.cache_name}'")

        for chunk in chunks_for_deletion:
            # Use chunk_id as key (same as in add_chunks)
            key = chunk.chunk_id

            response = await self.client.delete(f"{self.base_url}/rest/v3/caches/{self.cache_name}/entries/{key}")

            if response.status_code not in [200, 204, 404]:
                # 404 is acceptable - chunk may not exist
                log.error(f"Failed to delete chunk {key}: {response.status_code} - {response.text}")
                response.raise_for_status()

        log.info(f"Successfully deleted {len(chunks_for_deletion)} chunks")

    async def query_vector(
        self, embedding: NDArray, k: int, score_threshold: float, filters: Filter | None = None
    ) -> QueryChunksResponse:
        """
        Perform vector similarity search using Infinispan's vector search capabilities.

        Args:
            embedding: Query embedding vector as NumPy array
            k: Number of results to return
            score_threshold: Minimum similarity score threshold
            filters: Optional filters to apply to the search

        Returns:
            QueryChunksResponse with matching chunks and scores
        """
        # Filters are not yet implemented for Infinispan provider
        if filters is not None:
            raise NotImplementedError("Infinispan provider does not yet support native filtering")

        log.info(f"Performing vector similarity search in cache '{self.cache_name}' with k={k}")

        # Convert embedding to list
        query_vector = embedding.tolist() if hasattr(embedding, "tolist") else list(embedding)

        # Build Ickle query for vector similarity search
        # Use the <-> operator with ~ for KNN search (top-k nearest neighbors)
        entity_name = f"VectorItem{self.embedding_dimension}"
        # Format: WHERE vector <-> [query] ~ k returns top-k nearest neighbors
        ickle_query = f"SELECT i, score(i) FROM {entity_name} i WHERE i.floatVector <-> {query_vector} ~ {k}"

        # Execute vector search query
        response = await self.client.post(
            f"{self.base_url}/rest/v3/caches/{self.cache_name}/_search",
            json={"query": ickle_query, "max_results": k, "query_mode": "INDEXED"},
            headers={"Content-Type": "application/json"},
        )

        if response.status_code != 200:
            log.error(f"Vector search query failed: {response.status_code} - {response.text}")
            response.raise_for_status()

        # Parse search results
        search_results = response.json()
        log.info(f"Vector search returned {search_results.get('hit_count', 0)} total results")

        chunks = []
        scores = []

        # Process hits from the search response
        hits = search_results.get("hits", [])
        for hit in hits:
            # Extract the hit data - it may be nested under 'hit' or have a projection key
            hit = hit.get("hit", hit)

            hit_data = hit["*"]

            # Reconstruct EmbeddedChunk from the hit data
            # The hit contains the VectorItem fields: id, floatVector, text, metadata, chunkMetadata, embeddingModel
            chunk_id = hit_data.get("id")
            text = hit_data.get("text", "")
            float_vector = hit_data.get("floatVector", [])
            metadata_str = hit_data.get("metadata", "{}")
            chunk_metadata_str = hit_data.get("chunkMetadata", "{}")
            embedding_model = hit_data.get("embeddingModel", "unknown")

            if not chunk_id or not float_vector:
                log.warning(f"Skipping incomplete hit: {hit_data}")
                continue

            # Deserialize metadata
            try:
                metadata = json.loads(metadata_str) if metadata_str else {}
            except json.JSONDecodeError:
                log.warning(f"Failed to parse metadata for chunk {chunk_id}, using empty dict")
                metadata = {}

            # Deserialize chunk_metadata
            try:
                chunk_metadata = json.loads(chunk_metadata_str) if chunk_metadata_str else {}
            except json.JSONDecodeError:
                log.warning(f"Failed to parse chunk_metadata for chunk {chunk_id}, using empty dict")
                chunk_metadata = {}

            # Create EmbeddedChunk object
            chunk_dict = {
                "chunk_id": chunk_id,
                "document_id": chunk_metadata.get(
                    "document_id", chunk_id
                ),  # Get from chunk_metadata or use chunk_id as fallback
                "embedding": float_vector,
                "content": text,
                "metadata": metadata,
                "chunk_metadata": chunk_metadata,
                "embedding_model": embedding_model,
                "embedding_dimension": len(float_vector),
            }

            try:
                chunk = load_embedded_chunk_with_backward_compat(chunk_dict)
            except Exception as e:
                log.error(f"Failed to load chunk {chunk_id}: {e}")
                continue

            # Get score from hit (Infinispan returns computed similarity score)
            score = hit.get("score()", 1.0)

            # Filter by score threshold
            if score >= score_threshold:
                chunks.append(chunk)
                scores.append(score)

        log.info(f"Returning {len(chunks)} chunks after filtering by score_threshold={score_threshold}")
        return QueryChunksResponse(chunks=chunks, scores=scores)

    async def query_keyword(
        self,
        query_string: str,
        k: int,
        score_threshold: float,
        filters: Filter | None = None,
    ) -> QueryChunksResponse:
        """
        Perform full-text/keyword search using Infinispan Query DSL (Ickle).

        Args:
            query_string: The text query for keyword search
            k: Number of results to return
            score_threshold: Minimum similarity score threshold
            filters: Optional filters to apply to the search

        Returns:
            QueryChunksResponse with matching chunks and scores
        """
        # Infinispan provider does not yet support native filtering
        if filters is not None:
            raise NotImplementedError("Infinispan provider does not yet support native filtering")

        log.info(f"Performing keyword search in cache '{self.cache_name}' with query: {query_string}")

        # Build Ickle query to search the text field
        # The text field has @Keyword annotation, so it's indexed for full-text search
        entity_name = f"VectorItem{self.embedding_dimension}"
        # Escape single quotes to prevent query injection (similar to SQL injection prevention)
        escaped_query_string = query_string.replace("'", "''")
        ickle_query = f"SELECT i, score(i) FROM {entity_name} i WHERE text : '{escaped_query_string}' ~ 2"

        # Execute search query
        response = await self.client.post(
            f"{self.base_url}/rest/v3/caches/{self.cache_name}/_search",
            json={"query": ickle_query, "max_results": k, "query_mode": "INDEXED"},
            headers={"Content-Type": "application/json"},
        )

        if response.status_code != 200:
            log.error(f"Search query failed: {response.status_code} - {response.text}")
            response.raise_for_status()

        # Parse search results
        search_results = response.json()
        log.info(f"Search returned {search_results.get('hit_count', 0)} total results")

        chunks = []
        scores = []

        # Process hits from the search response
        hits = search_results.get("hits", [])
        for hit in hits:
            # Extract the hit data - it may be nested under 'hit' or have a projection key
            hit = hit.get("hit", hit)

            hit_data = hit["*"]

            # Reconstruct EmbeddedChunk from the hit data
            # The hit contains the VectorItem fields: id, floatVector, text, metadata, chunkMetadata, embeddingModel
            chunk_id = hit_data.get("id")
            text = hit_data.get("text", "")
            float_vector = hit_data.get("floatVector", [])
            metadata_str = hit_data.get("metadata", "{}")
            chunk_metadata_str = hit_data.get("chunkMetadata", "{}")
            embedding_model = hit_data.get("embeddingModel", "unknown")

            if not chunk_id or not float_vector:
                log.warning(f"Skipping incomplete hit: {hit_data}")
                continue

            # Deserialize metadata
            try:
                metadata = json.loads(metadata_str) if metadata_str else {}
            except json.JSONDecodeError:
                log.warning(f"Failed to parse metadata for chunk {chunk_id}, using empty dict")
                metadata = {}

            # Deserialize chunk_metadata
            try:
                chunk_metadata = json.loads(chunk_metadata_str) if chunk_metadata_str else {}
            except json.JSONDecodeError:
                log.warning(f"Failed to parse chunk_metadata for chunk {chunk_id}, using empty dict")
                chunk_metadata = {}

            # Create EmbeddedChunk object
            chunk_dict = {
                "chunk_id": chunk_id,
                "document_id": chunk_metadata.get("document_id", chunk_id),
                "embedding": float_vector,
                "content": text,
                "metadata": metadata,
                "chunk_metadata": chunk_metadata,
                "embedding_model": embedding_model,
                "embedding_dimension": len(float_vector),
            }

            try:
                chunk = load_embedded_chunk_with_backward_compat(chunk_dict)
                log.info(
                    f"Hit content - ID: {chunk_id}, Text: {text[:100]}{'...' if len(text) > 100 else ''}, Vector dim: {len(float_vector)}"
                )
            except Exception as e:
                log.error(f"Failed to load chunk {chunk_id}: {e}")
                continue

            # Get score from hit (Infinispan returns score for keyword search)
            # Default to 1.0 if score not available
            score = hit.get("score()", 1.0)

            # Filter by score threshold
            if score >= score_threshold:
                chunks.append(chunk)
                scores.append(score)

        log.info(f"Returning {len(chunks)} chunks after filtering by score_threshold={score_threshold}")
        return QueryChunksResponse(chunks=chunks, scores=scores)

    async def query_hybrid(
        self,
        embedding: NDArray,
        query_string: str,
        k: int,
        score_threshold: float,
        reranker_type: str,
        reranker_params: dict[str, Any] | None = None,
        filters: Filter | None = None,
    ) -> QueryChunksResponse:
        """
        Hybrid search combining vector similarity and keyword search using configurable reranking.

        This method is FULLY IMPLEMENTED and uses the standard pattern with WeightedInMemoryAggregator.
        It combines results from both query_vector() and query_keyword() methods.

        Args:
            embedding: The query embedding vector
            query_string: The text query for keyword search
            k: Number of results to return
            score_threshold: Minimum similarity score threshold
            reranker_type: Type of reranker to use ("rrf" or "weighted")
            reranker_params: Parameters for the reranker
            filters: Optional filters to apply to the search

        Returns:
            QueryChunksResponse with combined and reranked results
        """
        # Infinispan provider does not yet support native filtering
        if filters is not None:
            raise NotImplementedError("Infinispan provider does not yet support native filtering")

        if reranker_params is None:
            reranker_params = {}

        # Get results from both search methods
        vector_response = await self.query_vector(embedding, k, score_threshold)
        keyword_response = await self.query_keyword(query_string, k, score_threshold)

        # Convert responses to score dictionaries using chunk_id
        vector_scores = {
            chunk.chunk_id: float(score)
            for chunk, score in zip(vector_response.chunks, vector_response.scores, strict=False)
        }
        keyword_scores = {
            chunk.chunk_id: float(score)
            for chunk, score in zip(keyword_response.chunks, keyword_response.scores, strict=False)
        }

        # Combine scores using the reranking utility
        combined_scores = WeightedInMemoryAggregator.combine_search_results(
            vector_scores, keyword_scores, reranker_type, reranker_params
        )

        # Efficient top-k selection
        top_k_items = heapq.nlargest(k, combined_scores.items(), key=lambda x: x[1])

        # Filter by score threshold
        filtered_items = [(doc_id, score) for doc_id, score in top_k_items if score >= score_threshold]

        # Create a map of chunk_id to chunk for both responses
        chunk_map = {c.chunk_id: c for c in vector_response.chunks + keyword_response.chunks}

        # Use the map to look up chunks by their IDs
        chunks = []
        scores = []
        for doc_id, score in filtered_items:
            if doc_id in chunk_map:
                chunks.append(chunk_map[doc_id])
                scores.append(score)

        return QueryChunksResponse(chunks=chunks, scores=scores)

    async def delete(self):
        """
        Delete the entire Infinispan cache.

        Uses Infinispan REST API v3:
        - DELETE /rest/v3/caches/{cacheName}
        """
        log.info(f"Deleting cache '{self.cache_name}'")

        response = await self.client.delete(f"{self.base_url}/rest/v3/caches/{self.cache_name}")

        if response.status_code not in [200, 204]:
            log.error(f"Failed to delete cache '{self.cache_name}': {response.status_code} - {response.text}")
            response.raise_for_status()

        log.info(f"Cache '{self.cache_name}' deleted successfully")


class InfinispanVectorIOAdapter(OpenAIVectorStoreMixin, VectorIO, VectorStoresProtocolPrivate):
    """
    Infinispan adapter for vector store operations.
    Uses httpx.AsyncClient for HTTP REST API communication with Infinispan server.
    """

    def __init__(
        self,
        config: InfinispanVectorIOConfig,
        inference_api: Inference,
        files_api: Files | None = None,
        file_processor_api: FileProcessors | None = None,
    ) -> None:
        super().__init__(
            inference_api=inference_api, files_api=files_api, kvstore=None, file_processor_api=file_processor_api
        )
        log.info(f"Initializing InfinispanVectorIOAdapter with config: {config}")
        self.config = config
        self.client: httpx.AsyncClient | None = None
        self.cache: dict[str, VectorStoreWithIndex] = {}
        self.vector_store_table = None

    async def initialize(self) -> None:
        """
        Initialize the Infinispan adapter:
        1. Setup HTTP client with authentication
        2. Initialize KVStore for metadata persistence
        3. Load existing vector stores from KVStore
        """
        # Initialize KVStore for metadata persistence
        self.kvstore = await kvstore_impl(self.config.persistence)

        # Setup HTTP client with authentication
        auth: httpx.BasicAuth | httpx.DigestAuth | None = None
        if self.config.username and self.config.password:
            # Extract password from SecretStr if needed
            password = self.config.password.get_secret_value() if self.config.password else None
            if password:
                if self.config.auth_mechanism == "basic":
                    auth = httpx.BasicAuth(username=self.config.username, password=password)
                elif self.config.auth_mechanism == "digest":
                    auth = httpx.DigestAuth(username=self.config.username, password=password)
                else:
                    log.warning(f"Unknown auth mechanism: {self.config.auth_mechanism}, using BasicAuth")
                    auth = httpx.BasicAuth(username=self.config.username, password=password)

        # Create async HTTP client
        verify_tls = self.config.verify_tls if self.config.use_https else False
        self.client = httpx.AsyncClient(
            auth=auth,
            verify=verify_tls,
            timeout=30.0,
        )

        if self.config.use_https and not self.config.verify_tls:
            log.warning(
                "TLS certificate verification is disabled. "
                "This should only be used for development/testing with self-signed certificates. "
                "DO NOT use in production environments."
            )

        log.info(f"Connected to Infinispan server at: {str(self.config.url)}")

        # Load existing vector stores from KVStore
        await self._load_vector_stores_from_kvstore()

        # Initialize OpenAI vector stores
        self.openai_vector_stores = await self._load_openai_vector_stores()

    async def _load_vector_stores_from_kvstore(self) -> None:
        """
        Load existing vector stores from KVStore into the in-memory cache.
        """
        if self.kvstore is None:
            return

        # Scan KVStore for all vector stores with our prefix
        try:
            log.info("Loading vector stores from KVStore...")

            # Use keys_in_range to get all keys with the VECTOR_DBS_PREFIX
            # The range is from the prefix to the prefix with a high Unicode character
            start_key = VECTOR_DBS_PREFIX
            end_key = VECTOR_DBS_PREFIX + "\xff"

            keys = await self.kvstore.keys_in_range(start_key, end_key)
            log.info(f"Found {len(keys)} vector stores in KVStore")

            for key in keys:
                try:
                    # Get the vector store data
                    vector_store_data = await self.kvstore.get(key)
                    if not vector_store_data:
                        log.warning(f"Empty data for key {key}, skipping")
                        continue

                    # Deserialize the vector store
                    vector_store = VectorStore.model_validate_json(vector_store_data)

                    # Create the InfinispanIndex
                    assert self.client is not None, "HTTP client must be initialized"
                    index = InfinispanIndex(
                        client=self.client,
                        cache_name=vector_store.identifier,
                        base_url=str(self.config.url),
                        embedding_dimension=vector_store.embedding_dimension,
                        kvstore=self.kvstore,
                    )

                    # Note: We don't call index.initialize() here because the cache
                    # should already exist. If it doesn't, it will be created when
                    # the vector store is actually used.

                    # Create VectorStoreWithIndex and add to cache
                    vector_store_with_index = VectorStoreWithIndex(vector_store, index, self.inference_api)
                    self.cache[vector_store.identifier] = vector_store_with_index
                    log.info(f"Loaded vector store: {vector_store.identifier}")

                except Exception as e:
                    log.error(f"Failed to load vector store from key {key}: {e}")
                    continue

        except Exception as e:
            log.warning(f"Failed to load vector stores from KVStore: {e}")

    async def shutdown(self) -> None:
        """Clean up resources including HTTP client and mixin resources."""
        # Clean up mixin resources (file batch tasks)
        await super().shutdown()

        # Close HTTP client
        if self.client:
            await self.client.aclose()

    async def register_vector_store(self, vector_store: VectorStore) -> None:
        """
        Register a new vector store by creating an Infinispan cache and storing metadata.

        Args:
            vector_store: VectorStore object with configuration
        """
        # Persist vector DB metadata in the KV store
        if self.kvstore is None:
            raise RuntimeError("KVStore not initialized. Call initialize() before registering vector stores.")

        # Save to kvstore for persistence
        key = f"{VECTOR_DBS_PREFIX}{vector_store.identifier}"
        await self.kvstore.set(key=key, value=vector_store.model_dump_json())

        # Create Infinispan index
        assert self.client is not None, "HTTP client must be initialized"
        index = InfinispanIndex(
            client=self.client,
            cache_name=vector_store.identifier,
            base_url=str(self.config.url),
            embedding_dimension=vector_store.embedding_dimension,
            kvstore=self.kvstore,
        )

        # Initialize the cache in Infinispan
        await index.initialize()

        # Create VectorStoreWithIndex and add to cache
        vector_store_with_index = VectorStoreWithIndex(vector_store, index, self.inference_api)
        self.cache[vector_store.identifier] = vector_store_with_index
        log.info(f"Registered vector store: {vector_store.identifier}")

        # Persist to KVStore
        if self.kvstore:
            key = f"{VECTOR_DBS_PREFIX}{vector_store.identifier}"
            await self.kvstore.set(key, vector_store.model_dump_json())

    async def unregister_vector_store(self, vector_store_id: str) -> None:
        """
        Delete a vector store from both Infinispan and KVStore.

        Args:
            vector_store_id: Identifier of the vector store to delete
        """
        if vector_store_id not in self.cache:
            log.debug(f"Vector DB {vector_store_id} not found in cache, skipping deletion")
            return

        # Delete from Infinispan
        await self.cache[vector_store_id].index.delete()

        # Remove from KVStore
        if self.kvstore:
            key = f"{VECTOR_DBS_PREFIX}{vector_store_id}"
            await self.kvstore.delete(key)

        # Remove from cache
        del self.cache[vector_store_id]

    async def insert_chunks(self, request: InsertChunksRequest) -> None:
        """
        Insert chunks into a vector store.

        Args:
            request: InsertChunksRequest containing vector_store_id, chunks, and optional ttl_seconds
        """
        index = await self._get_and_cache_vector_store_index(request.vector_store_id)
        if index is None:
            raise ValueError(f"Vector DB {request.vector_store_id} not found in Infinispan")

        await index.insert_chunks(request)

    async def query_chunks(self, request: QueryChunksRequest) -> QueryChunksResponse:
        """
        Query chunks from a vector store.

        Args:
            request: QueryChunksRequest containing vector_store_id, query, and optional params

        Returns:
            QueryChunksResponse with matching chunks and scores
        """
        index = await self._get_and_cache_vector_store_index(request.vector_store_id)

        if index is None:
            raise ValueError(f"Vector DB {request.vector_store_id} not found in Infinispan")

        return await index.query_chunks(request)

    async def delete_chunks(self, request: DeleteChunksRequest) -> None:
        """
        Delete specific chunks from a vector store.

        Args:
            request: DeleteChunksRequest containing vector_store_id and chunks to delete
        """
        index = await self._get_and_cache_vector_store_index(request.vector_store_id)
        if not index:
            raise ValueError(f"Vector DB {request.vector_store_id} not found")

        await index.index.delete_chunks(request.chunks)

    async def _get_and_cache_vector_store_index(self, vector_store_id: str) -> VectorStoreWithIndex:
        """
        Get a vector store index from cache or load from KVStore.

        Args:
            vector_store_id: Identifier of the vector store

        Returns:
            VectorStoreWithIndex object

        Raises:
            ValueError: If vector store not found
        """
        # Check cache first
        if vector_store_id in self.cache:
            return self.cache[vector_store_id]

        # Try to load from KVStore
        if self.kvstore is None:
            raise RuntimeError("KVStore not initialized. Call initialize() before using vector stores.")

        key = f"{VECTOR_DBS_PREFIX}{vector_store_id}"
        vector_store_data = await self.kvstore.get(key)
        if not vector_store_data:
            raise ValueError(f"Vector DB {vector_store_id} not found in OGX")

        # Deserialize vector store metadata
        vector_store = VectorStore.model_validate_json(vector_store_data)

        # Create index for the Infinispan cache
        assert self.client is not None, "HTTP client must be initialized"
        index = InfinispanIndex(
            client=self.client,
            cache_name=vector_store_id,
            base_url=str(self.config.url),
            embedding_dimension=vector_store.embedding_dimension,
            kvstore=self.kvstore,
        )

        # Create VectorStoreWithIndex and add to cache
        vector_store_with_index: VectorStoreWithIndex = VectorStoreWithIndex(vector_store, index, self.inference_api)
        self.cache[vector_store_id] = vector_store_with_index

        return vector_store_with_index
