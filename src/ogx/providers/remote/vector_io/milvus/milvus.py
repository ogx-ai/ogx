# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import asyncio
import heapq
import os
from typing import Any

from numpy.typing import NDArray
from pymilvus import AnnSearchRequest, DataType, Function, FunctionType, MilvusClient, RRFRanker, WeightedRanker

from ogx.core.storage.kvstore import kvstore_impl
from ogx.log import get_logger
from ogx.providers.inline.vector_io.milvus import MilvusVectorIOConfig as InlineMilvusVectorIOConfig
from ogx.providers.utils.memory.openai_vector_store_mixin import OpenAIVectorStoreMixin
from ogx.providers.utils.memory.vector_store import (
    RERANKER_TYPE_WEIGHTED,
    ChunkForDeletion,
    EmbeddingIndex,
    VectorStoreWithIndex,
)
from ogx.providers.utils.vector_io.vector_utils import (
    WeightedInMemoryAggregator,
    load_embedded_chunk_with_backward_compat,
    sanitize_collection_name,
)
from ogx_api import (
    ComparisonFilter,
    CompoundFilter,
    DeleteChunksRequest,
    EmbeddedChunk,
    FileProcessors,
    Files,
    Filter,
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

from .config import MilvusVectorIOConfig as RemoteMilvusVectorIOConfig

logger = get_logger(name=__name__, category="vector_io::milvus")


def _fmt(v: Any) -> str:
    return f'"{v}"' if isinstance(v, str) else str(v)


_MILVUS_OPS: dict[str, str] = {
    "gt": ">",
    "gte": ">=",
    "lt": "<",
    "lte": "<=",
}

VERSION = "v3"
VECTOR_DBS_PREFIX = f"vector_stores:milvus:{VERSION}::"
VECTOR_INDEX_PREFIX = f"vector_index:milvus:{VERSION}::"
OPENAI_VECTOR_STORES_PREFIX = f"openai_vector_stores:milvus:{VERSION}::"
OPENAI_VECTOR_STORES_FILES_PREFIX = f"openai_vector_stores_files:milvus:{VERSION}::"
OPENAI_VECTOR_STORES_FILES_CONTENTS_PREFIX = f"openai_vector_stores_files_contents:milvus:{VERSION}::"


class MilvusIndex(EmbeddingIndex):
    """Embedding index backed by a Milvus collection."""

    def __init__(
        self,
        client: MilvusClient,
        collection_name: str,
        consistency_level: str = "Strong",
        kvstore: KVStore | None = None,
        use_native_hybrid: bool = False,
    ):
        self.client = client
        self.collection_name = sanitize_collection_name(collection_name)
        self.consistency_level = consistency_level
        self.kvstore = kvstore
        self.use_native_hybrid = use_native_hybrid

    async def initialize(self):
        # MilvusIndex does not require explicit initialization
        # TODO: could move collection creation into initialization but it is not really necessary
        pass

    async def delete(self):
        if await asyncio.to_thread(self.client.has_collection, self.collection_name):
            await asyncio.to_thread(self.client.drop_collection, collection_name=self.collection_name)

    async def add_chunks(self, chunks: list[EmbeddedChunk]):
        if not chunks:
            return

        if not await asyncio.to_thread(self.client.has_collection, self.collection_name):
            logger.info("Creating new collection with nullable sparse field", collection_name=self.collection_name)
            # Create schema for vector search
            schema = self.client.create_schema()
            schema.add_field(field_name="chunk_id", datatype=DataType.VARCHAR, is_primary=True, max_length=100)
            schema.add_field(
                field_name="content",
                datatype=DataType.VARCHAR,
                max_length=65535,
                enable_analyzer=True,  # Enable text analysis for BM25
            )
            schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=len(chunks[0].embedding))
            schema.add_field(field_name="chunk_content", datatype=DataType.JSON)
            # Add sparse vector field for BM25 (required by the function)
            schema.add_field(field_name="sparse", datatype=DataType.SPARSE_FLOAT_VECTOR)

            # Create indexes
            index_params = self.client.prepare_index_params()
            index_params.add_index(field_name="vector", index_type="FLAT", metric_type="COSINE")
            # Add index for sparse field (required by BM25 function)
            index_params.add_index(field_name="sparse", index_type="SPARSE_INVERTED_INDEX", metric_type="BM25")

            # Add BM25 function for full-text search
            bm25_function = Function(
                name="text_bm25_emb",
                input_field_names=["content"],
                output_field_names=["sparse"],
                function_type=FunctionType.BM25,
            )
            schema.add_function(bm25_function)

            await asyncio.to_thread(
                self.client.create_collection,
                self.collection_name,
                schema=schema,
                index_params=index_params,
                consistency_level=self.consistency_level,
            )

        data = []
        for chunk in chunks:
            data.append(
                {
                    "chunk_id": chunk.chunk_id,
                    "content": chunk.content,
                    "vector": chunk.embedding,  # Already a list[float]
                    "chunk_content": chunk.model_dump(),
                    # sparse field will be handled by BM25 function automatically
                }
            )
        try:
            await asyncio.to_thread(self.client.insert, self.collection_name, data=data)
        except Exception as e:
            logger.error(
                "Error inserting chunks into Milvus collection", collection_name=self.collection_name, error=str(e)
            )
            raise e

    def _translate_filters(self, filters: Filter | None) -> str | None:
        """Translate OpenAI-compatible filters to Milvus expression format.

        Args:
            filters: The filter to translate (ComparisonFilter or CompoundFilter)

        Returns:
            A Milvus expression string or None if no filters
        """
        if filters is None:
            return None

        return self._translate_single_filter(filters)

    def _translate_single_filter(self, filter_obj: Filter) -> str:
        """Translate a single filter to Milvus expression."""
        if isinstance(filter_obj, ComparisonFilter):
            return self._translate_comparison_filter(filter_obj)
        elif isinstance(filter_obj, CompoundFilter):
            return self._translate_compound_filter(filter_obj)
        else:
            raise ValueError(f"Unknown filter type: {type(filter_obj)}")

    def _translate_comparison_filter(self, filter_obj: ComparisonFilter) -> str:
        """Translate a comparison filter to Milvus expression.

        Milvus uses JSONPath-like expressions to access metadata fields.
        The metadata is stored in the chunk_content JSON field.
        """
        key, value, op_type = filter_obj.key, filter_obj.value, filter_obj.type
        json_path = f"chunk_content['metadata']['{key}']"

        if op_type in ("eq", "ne"):
            sym = "==" if op_type == "eq" else "!="
            return f"{json_path} {sym} {_fmt(value)}"
        elif op_type in _MILVUS_OPS:
            return f"{json_path} {_MILVUS_OPS[op_type]} {value}"
        elif op_type in ("in", "nin"):
            if not isinstance(value, list):
                raise ValueError(f"'{op_type}' filter requires a list value, got {type(value)}")
            formatted = [_fmt(v) for v in value]
            kw = "not in" if op_type == "nin" else "in"
            return f"{json_path} {kw} [{', '.join(formatted)}]"
        else:
            raise ValueError(f"Unsupported comparison operator: {op_type}")

    def _translate_compound_filter(self, filter_obj: CompoundFilter) -> str:
        """Translate a compound filter (and/or) to Milvus expression."""
        if not filter_obj.filters:
            return ""

        clauses = []
        for sub_filter in filter_obj.filters:
            clause = self._translate_single_filter(sub_filter)
            if clause:
                clauses.append(f"({clause})")

        if not clauses:
            return ""

        operator = " and " if filter_obj.type == "and" else " or "
        return operator.join(clauses)

    async def query_vector(
        self, embedding: NDArray, k: int, score_threshold: float, filters: Any = None
    ) -> QueryChunksResponse:
        # Translate filters to Milvus expression format
        filter_expr = self._translate_filters(filters) if filters else None

        search_kwargs = {
            "collection_name": self.collection_name,
            "data": [embedding],
            "anns_field": "vector",
            "limit": k,
            "output_fields": ["*"],
        }

        # Only apply radius threshold if score_threshold is meaningful
        # For cosine similarity, distance ranges from 0 (identical) to 2 (opposite)
        if score_threshold > 0:
            search_kwargs["search_params"] = {"params": {"radius": score_threshold}}

        if filter_expr:
            search_kwargs["filter"] = filter_expr

        search_res = await asyncio.to_thread(self.client.search, **search_kwargs)
        chunks = [load_embedded_chunk_with_backward_compat(res["entity"]["chunk_content"]) for res in search_res[0]]
        scores = [res["distance"] for res in search_res[0]]
        return QueryChunksResponse(chunks=chunks, scores=scores)

    async def query_keyword(
        self, query_string: str, k: int, score_threshold: float, filters: Filter | None = None
    ) -> QueryChunksResponse:
        """
        Perform BM25-based keyword search using Milvus's built-in full-text search.
        """
        try:
            # Translate filters to Milvus expression format
            filter_expr = self._translate_filters(filters) if filters else None

            search_kwargs = {
                "collection_name": self.collection_name,
                "data": [query_string],  # Raw text query
                "anns_field": "sparse",  # Use sparse field for BM25
                "output_fields": ["chunk_content"],  # Output the chunk content
                "limit": k,
                "search_params": {
                    "params": {
                        "drop_ratio_search": 0.2,  # Ignore low-importance terms
                    }
                },
            }

            if filter_expr:
                search_kwargs["filter"] = filter_expr

            # Use Milvus's built-in BM25 search
            search_res = await asyncio.to_thread(self.client.search, **search_kwargs)

            chunks = []
            scores = []
            for res in search_res[0]:
                chunk = load_embedded_chunk_with_backward_compat(res["entity"]["chunk_content"])
                chunks.append(chunk)
                scores.append(res["distance"])  # BM25 score from Milvus

            # Filter by score threshold
            filtered_chunks = [chunk for chunk, score in zip(chunks, scores, strict=False) if score >= score_threshold]
            filtered_scores = [score for score in scores if score >= score_threshold]

            return QueryChunksResponse(chunks=filtered_chunks, scores=filtered_scores)

        except Exception as e:
            logger.error("Error performing BM25 search", error=str(e))
            # Fallback to simple text search
            return await self._fallback_keyword_search(query_string, k, score_threshold)

    async def _fallback_keyword_search(self, query_string: str, k: int, score_threshold: float) -> QueryChunksResponse:
        """
        Fallback to simple text search when BM25 search is not available.
        """
        # Simple text search using content field
        search_res = await asyncio.to_thread(
            self.client.query,
            collection_name=self.collection_name,
            filter='content like "%{content}%"',
            filter_params={"content": query_string},
            output_fields=["*"],
            limit=k,
        )
        chunks = [load_embedded_chunk_with_backward_compat(res["chunk_content"]) for res in search_res]
        scores = [1.0] * len(chunks)  # Simple binary score for text search
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
        if self.use_native_hybrid:
            return await self._query_hybrid_native(
                embedding, query_string, k, score_threshold, reranker_type, reranker_params, filters
            )
        return await self._query_hybrid_in_memory(
            embedding, query_string, k, score_threshold, reranker_type, reranker_params, filters
        )

    async def _query_hybrid_native(
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
        Hybrid search using Milvus's native hybrid search capabilities.

        Uses Milvus's hybrid_search method which combines vector search and
        BM25 search server-side with configurable reranking strategies.
        """
        search_requests = []

        search_requests.append(
            AnnSearchRequest(data=[embedding.tolist()], anns_field="vector", param={"nprobe": 10}, limit=k)
        )

        search_requests.append(
            AnnSearchRequest(data=[query_string], anns_field="sparse", param={"drop_ratio_search": 0.2}, limit=k)
        )

        if reranker_type == RERANKER_TYPE_WEIGHTED:
            alpha = (reranker_params or {}).get("alpha", 0.5)
            rerank = WeightedRanker(alpha, 1 - alpha)
        else:
            impact_factor = (reranker_params or {}).get("impact_factor", 60.0)
            rerank = RRFRanker(impact_factor)

        filter_expr = self._translate_filters(filters) if filters else None

        search_kwargs = {
            "collection_name": self.collection_name,
            "reqs": search_requests,
            "ranker": rerank,
            "limit": k,
            "output_fields": ["chunk_content"],
        }

        if filter_expr:
            search_kwargs["filter"] = filter_expr

        search_res = await asyncio.to_thread(self.client.hybrid_search, **search_kwargs)

        chunks = []
        scores = []
        for res in search_res[0]:
            chunk = load_embedded_chunk_with_backward_compat(res["entity"]["chunk_content"])
            chunks.append(chunk)
            scores.append(res["distance"])

        return QueryChunksResponse(chunks=chunks, scores=scores)

    async def _query_hybrid_in_memory(
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
        Hybrid search combining vector similarity and keyword search using in-memory aggregation.

        Calls the standalone query_vector() and query_keyword() methods and combines
        results using WeightedInMemoryAggregator, consistent with pgvector, sqlite_vec,
        chroma, and oci providers.
        """
        if reranker_params is None:
            reranker_params = {}

        vector_response = await self.query_vector(embedding, k, score_threshold, filters)
        keyword_response = await self.query_keyword(query_string, k, score_threshold, filters)

        vector_scores = {
            chunk.chunk_id: score for chunk, score in zip(vector_response.chunks, vector_response.scores, strict=False)
        }
        keyword_scores = {
            chunk.chunk_id: score
            for chunk, score in zip(keyword_response.chunks, keyword_response.scores, strict=False)
        }

        combined_scores = WeightedInMemoryAggregator.combine_search_results(
            vector_scores, keyword_scores, reranker_type, reranker_params
        )

        top_k_items = heapq.nlargest(k, combined_scores.items(), key=lambda x: x[1])
        filtered_items = [(doc_id, score) for doc_id, score in top_k_items if score >= score_threshold]

        chunk_map = {c.chunk_id: c for c in vector_response.chunks + keyword_response.chunks}

        chunks = []
        scores = []
        for doc_id, score in filtered_items:
            if doc_id in chunk_map:
                chunks.append(chunk_map[doc_id])
                scores.append(score)

        return QueryChunksResponse(chunks=chunks, scores=scores)

    async def delete_chunks(self, chunks_for_deletion: list[ChunkForDeletion]) -> None:
        """Remove a chunk from the Milvus collection."""
        chunk_ids = [c.chunk_id for c in chunks_for_deletion]
        try:
            # Use IN clause with square brackets and single quotes for VARCHAR field
            chunk_ids_str = ", ".join(f"'{chunk_id}'" for chunk_id in chunk_ids)
            await asyncio.to_thread(
                self.client.delete, collection_name=self.collection_name, filter=f"chunk_id in [{chunk_ids_str}]"
            )
        except Exception as e:
            logger.error(
                "Error deleting chunks from Milvus collection", collection_name=self.collection_name, error=str(e)
            )
            raise


class MilvusVectorIOAdapter(OpenAIVectorStoreMixin, VectorIO, VectorStoresProtocolPrivate):
    """Vector I/O adapter for remote Milvus instances."""

    def __init__(
        self,
        config: RemoteMilvusVectorIOConfig | InlineMilvusVectorIOConfig,
        inference_api: Inference,
        files_api: Files | None,
        file_processor_api: FileProcessors | None = None,
    ) -> None:
        super().__init__(
            inference_api=inference_api, files_api=files_api, kvstore=None, file_processor_api=file_processor_api
        )
        self.config = config
        self.cache = {}
        self.client = None
        self.vector_store_table = None
        self.metadata_collection_name = "openai_vector_stores_metadata"

    async def initialize(self) -> None:
        self.kvstore = await kvstore_impl(self.config.persistence)
        start_key = VECTOR_DBS_PREFIX
        end_key = f"{VECTOR_DBS_PREFIX}\xff"
        stored_vector_stores = await self.kvstore.values_in_range(start_key, end_key)

        use_native_hybrid = isinstance(self.config, RemoteMilvusVectorIOConfig)
        for vector_store_data in stored_vector_stores:
            vector_store = VectorStore.model_validate_json(vector_store_data)
            index = VectorStoreWithIndex(
                vector_store,
                index=MilvusIndex(
                    client=self.client,
                    collection_name=vector_store.identifier,
                    consistency_level=self.config.consistency_level,
                    kvstore=self.kvstore,
                    use_native_hybrid=use_native_hybrid,
                ),
                inference_api=self.inference_api,
            )
            self.cache[vector_store.identifier] = index
        if isinstance(self.config, RemoteMilvusVectorIOConfig):
            logger.info("Connecting to Milvus server at", uri=self.config.uri)
            self.client = MilvusClient(**self.config.model_dump(exclude_none=True))
        else:
            logger.info("Connecting to Milvus Lite at", db_path=self.config.db_path)
            uri = os.path.expanduser(self.config.db_path)
            self.client = MilvusClient(uri=uri)

        # Load existing OpenAI vector stores into the in-memory cache
        await self.initialize_openai_vector_stores()

    async def shutdown(self) -> None:
        self.client.close()
        # Clean up mixin resources (file batch tasks)
        await super().shutdown()

    async def register_vector_store(self, vector_store: VectorStore) -> None:
        use_native_hybrid = isinstance(self.config, RemoteMilvusVectorIOConfig)
        if isinstance(self.config, RemoteMilvusVectorIOConfig):
            consistency_level = self.config.consistency_level
        else:
            consistency_level = "Strong"
        index = VectorStoreWithIndex(
            vector_store=vector_store,
            index=MilvusIndex(
                self.client,
                vector_store.identifier,
                consistency_level=consistency_level,
                use_native_hybrid=use_native_hybrid,
            ),
            inference_api=self.inference_api,
        )

        self.cache[vector_store.identifier] = index

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
        use_native_hybrid = isinstance(self.config, RemoteMilvusVectorIOConfig)
        index = VectorStoreWithIndex(
            vector_store=vector_store,
            index=MilvusIndex(
                client=self.client,
                collection_name=vector_store.identifier,
                kvstore=self.kvstore,
                use_native_hybrid=use_native_hybrid,
            ),
            inference_api=self.inference_api,
        )
        self.cache[vector_store_id] = index
        return index

    async def unregister_vector_store(self, vector_store_id: str) -> None:
        if vector_store_id in self.cache:
            await self.cache[vector_store_id].index.delete()
            del self.cache[vector_store_id]

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
        """Delete a chunk from a milvus vector store."""
        index = await self._get_and_cache_vector_store_index(request.vector_store_id)
        if not index:
            raise VectorStoreNotFoundError(request.vector_store_id)

        await index.index.delete_chunks(request.chunks)
