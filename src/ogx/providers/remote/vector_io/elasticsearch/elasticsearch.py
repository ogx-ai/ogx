# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

from elasticsearch import ApiError, AsyncElasticsearch
from elasticsearch.helpers import async_bulk
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

from .config import ElasticsearchVectorIOConfig

log = get_logger(name=__name__, category="vector_io::elasticsearch")

# KV store prefixes for vector databases
VERSION = "v3"
VECTOR_DBS_PREFIX = f"vector_stores:elasticsearch:{VERSION}::"
VECTOR_INDEX_PREFIX = f"vector_index:elasticsearch:{VERSION}::"
OPENAI_VECTOR_STORES_PREFIX = f"openai_vector_stores:elasticsearch:{VERSION}::"
OPENAI_VECTOR_STORES_FILES_PREFIX = f"openai_vector_stores_files:elasticsearch:{VERSION}::"
OPENAI_VECTOR_STORES_FILES_CONTENTS_PREFIX = f"openai_vector_stores_files_contents:elasticsearch:{VERSION}::"


class ElasticsearchIndex(EmbeddingIndex):
    """Embedding index backed by an Elasticsearch index."""

    def __init__(self, client: AsyncElasticsearch, collection_name: str):
        self.client = client
        self.collection_name = collection_name

    # Check if the rerank_params contains the following structure:
    # {
    #   "retrievers": {
    #       "standard": {"weight": 0.7},
    #       "knn": {"weight": 0.3}
    #   }
    # }
    async def _is_rerank_linear_param_valid(self, value: dict) -> bool:
        """Validate linear reranker parameters structure."""
        try:
            retrievers = value.get("retrievers", {})
            return (
                isinstance(retrievers.get("standard"), dict)
                and isinstance(retrievers.get("knn"), dict)
                and "weight" in retrievers["standard"]
                and "weight" in retrievers["knn"]
            )
        except (AttributeError, TypeError):
            return False

    def _convert_to_linear_params(self, reranker_params: dict[str, Any]) -> dict[str, Any] | None:
        weights = reranker_params.get("weights")
        alpha = reranker_params.get("alpha")
        if weights is not None:
            vector_weight = weights.get("vector")
            keyword_weight = weights.get("keyword")
            if vector_weight is None or keyword_weight is None:
                log.warning("Elasticsearch linear retriever requires 'vector' and 'keyword' weights; ignoring weights.")
                return None
            total = vector_weight + keyword_weight
            if total == 0:
                log.warning(
                    "Elasticsearch linear retriever weights for 'vector' and 'keyword' sum to 0; ignoring weights."
                )
                return None
            if abs(total - 1.0) > 0.001:
                log.warning(
                    "Elasticsearch linear retriever uses normalized vector/keyword weights; "
                    "renormalizing provided weights."
                )
                vector_weight /= total
                keyword_weight /= total
        elif alpha is not None:
            vector_weight = alpha
            keyword_weight = 1 - alpha
        else:
            return None

        return {
            "retrievers": {
                "standard": {"weight": keyword_weight},
                "knn": {"weight": vector_weight},
            }
        }

    async def initialize(self) -> None:
        # Elasticsearch collections (indexes) are created on-demand in add_chunks
        # If the index does not exist, it will be created in add_chunks.
        pass

    async def add_chunks(self, chunks: list[EmbeddedChunk]):
        """Adds chunks to the Elasticsearch index."""
        if not chunks:
            return

        try:
            await self.client.indices.create(
                index=self.collection_name,
                body={
                    "mappings": {
                        "properties": {
                            "content": {"type": "text"},
                            "chunk_id": {"type": "keyword"},
                            "metadata": {"type": "object"},
                            "chunk_metadata": {"type": "object"},
                            "embedding": {"type": "dense_vector", "dims": len(chunks[0].embedding)},
                            "embedding_dimension": {"type": "integer"},
                            "embedding_model": {"type": "keyword"},
                        }
                    }
                },
            )
        except ApiError as e:
            if e.status_code != 400 or "resource_already_exists_exception" not in e.message:
                log.error(f"Error creating Elasticsearch index {self.collection_name}: {e}")
                raise

        actions = []
        for chunk in chunks:
            actions.append(
                {
                    "_op_type": "index",
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
            )

        try:
            successful_count, error_count = await async_bulk(
                client=self.client, actions=actions, timeout="300s", refresh=True, raise_on_error=False, stats_only=True
            )
            if error_count > 0:
                log.warning(
                    f"{error_count} out of {len(chunks)} documents failed to upload in Elasticsearch index {self.collection_name}"
                )

            log.info(f"Successfully added {successful_count} chunks to Elasticsearch index {self.collection_name}")
        except Exception as e:
            log.error(f"Error adding chunks to Elasticsearch index {self.collection_name}: {e}")
            raise

    async def delete_chunks(self, chunks_for_deletion: list[ChunkForDeletion]) -> None:
        """Remove a chunk from the Elasticsearch index."""

        actions = []
        for chunk in chunks_for_deletion:
            actions.append({"_op_type": "delete", "_index": self.collection_name, "_id": chunk.chunk_id})

        try:
            successful_count, error_count = await async_bulk(
                client=self.client, actions=actions, timeout="300s", refresh=True, raise_on_error=True, stats_only=True
            )
            if error_count > 0:
                log.warning(
                    f"{error_count} out of {len(chunks_for_deletion)} documents failed to be deleted in Elasticsearch index {self.collection_name}"
                )

            log.info(f"Successfully deleted {successful_count} chunks from Elasticsearch index {self.collection_name}")
        except Exception as e:
            log.error(f"Error deleting chunks from Elasticsearch index {self.collection_name}: {e}")
            raise

    async def _results_to_chunks(self, results: dict) -> QueryChunksResponse:
        """Convert search results to QueryChunksResponse."""

        chunks, scores = [], []
        for result in results.get("hits", {}).get("hits", []):
            try:
                source = result.get("_source", {})
                chunk = EmbeddedChunk(
                    content=source.get("content"),
                    chunk_id=result.get("_id"),
                    embedding=source.get("embedding", []),
                    embedding_dimension=source.get("embedding_dimension", len(source.get("embedding", []))),
                    embedding_model=source.get("embedding_model", "unknown"),
                    chunk_metadata=source.get("chunk_metadata", {}),
                    metadata=source.get("metadata", {}),
                )
            except Exception:
                log.exception("Failed to parse chunk")
                continue

            chunks.append(chunk)
            scores.append(result.get("_score"))

        return QueryChunksResponse(chunks=chunks, scores=scores)

    async def query_vector(
        self, embedding: NDArray, k: int, score_threshold: float, filters: Filter | None = None
    ) -> QueryChunksResponse:
        """Vector search using kNN."""
        if filters is not None:
            raise NotImplementedError("Elasticsearch provider does not yet support native filtering")

        try:
            results = await self.client.search(
                index=self.collection_name,
                query={"knn": {"field": "embedding", "query_vector": embedding.tolist(), "k": k}},
                min_score=score_threshold,
                size=k,
                source={"exclude_vectors": False},  # Retrieve the embedding
                ignore_unavailable=True,  # In case the index does not exist
            )
        except Exception as e:
            log.error(f"Error performing vector query on Elasticsearch index {self.collection_name}: {e}")
            raise

        return await self._results_to_chunks(results)

    async def query_keyword(
        self, query_string: str, k: int, score_threshold: float, filters: Filter | None = None
    ) -> QueryChunksResponse:
        """Keyword search using match query."""
        if filters is not None:
            raise NotImplementedError("Elasticsearch provider does not yet support native filtering")

        try:
            results = await self.client.search(
                index=self.collection_name,
                query={"match": {"content": {"query": query_string}}},
                min_score=score_threshold,
                size=k,
                source={"exclude_vectors": False},  # Retrieve the embedding
                ignore_unavailable=True,  # In case the index does not exist
            )
        except Exception as e:
            log.error(f"Error performing keyword query on Elasticsearch index {self.collection_name}: {e}")
            raise

        return await self._results_to_chunks(results)

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
        if filters is not None:
            raise NotImplementedError("Elasticsearch provider does not yet support native filtering")
        supported_retrievers = ["rrf", "linear"]
        original_reranker_type = reranker_type
        if reranker_type == "weighted":
            log.warning("Elasticsearch does not support 'weighted' reranker; using 'linear' retriever instead.")
            reranker_type = "linear"
        if reranker_type not in supported_retrievers:
            log.warning(
                f"Unsupported reranker type: {reranker_type}. Supported types are: {supported_retrievers}. "
                "Falling back to 'rrf'."
            )
            reranker_type = "rrf"

        retriever = {
            reranker_type: {
                "retrievers": [
                    {"retriever": {"standard": {"query": {"match": {"content": query_string}}}}},
                    {
                        "retriever": {
                            "knn": {
                                "field": "embedding",
                                "query_vector": embedding.tolist(),
                                "k": k,
                                "num_candidates": k,
                            }
                        }
                    },
                ]
            }
        }
        # Elasticsearch requires rank_window_size >= size for rrf/linear retrievers.
        retriever[reranker_type]["rank_window_size"] = k

        # Add reranker parameters if provided for RRF (e.g. rank_constant, rank_window_size, filter)
        # see https://www.elastic.co/docs/reference/elasticsearch/rest-apis/retrievers/rrf-retriever
        if reranker_type == "rrf" and reranker_params is not None:
            allowed_rrf_params = {"rank_constant", "rank_windows_size", "filter"}
            rrf_params = dict(reranker_params)
            if "impact_factor" in rrf_params:
                if "rank_constant" not in rrf_params:
                    rrf_params["rank_constant"] = rrf_params.pop("impact_factor")
                    log.warning("Elasticsearch RRF does not support impact_factor; mapping to rank_constant.")
                else:
                    rrf_params.pop("impact_factor")
                    log.warning("Elasticsearch RRF ignores impact_factor when rank_constant is provided.")
            if "rank_window_size" not in rrf_params and "rank_windows_size" in rrf_params:
                rrf_params["rank_window_size"] = rrf_params.pop("rank_windows_size")
            extra_keys = set(rrf_params.keys()) - allowed_rrf_params
            if extra_keys:
                log.warning(f"Ignoring unsupported RRF parameters for Elasticsearch: {extra_keys}")
                for key in extra_keys:
                    rrf_params.pop(key, None)
            if rrf_params:
                retriever["rrf"].update(rrf_params)
        elif reranker_type == "linear" and reranker_params is not None:
            # Add reranker parameters (i.e. weights) for linear
            # see https://www.elastic.co/docs/reference/elasticsearch/rest-apis/retrievers/linear-retriever
            if await self._is_rerank_linear_param_valid(reranker_params) is False:
                converted_params = self._convert_to_linear_params(reranker_params)
                if converted_params is None:
                    log.warning(
                        "Invalid linear reranker parameters for Elasticsearch; "
                        'expected {"retrievers": {"standard": {"weight": float}, "knn": {"weight": float}}}. '
                        "Ignoring provided parameters."
                    )
                else:
                    reranker_params = converted_params
            try:
                if await self._is_rerank_linear_param_valid(reranker_params):
                    retriever["linear"]["retrievers"][0].update(reranker_params["retrievers"]["standard"])
                    retriever["linear"]["retrievers"][1].update(reranker_params["retrievers"]["knn"])
            except Exception as e:
                log.error(f"Error updating linear retrievers parameters: {e}")
                raise
        elif reranker_type == "linear" and reranker_params is None and original_reranker_type == "weighted":
            converted_params = self._convert_to_linear_params({})
            if converted_params:
                retriever["linear"]["retrievers"][0].update(converted_params["retrievers"]["standard"])
                retriever["linear"]["retrievers"][1].update(converted_params["retrievers"]["knn"])
        try:
            results = await self.client.search(
                index=self.collection_name,
                size=k,
                retriever=retriever,
                min_score=score_threshold,
                source={"exclude_vectors": False},  # Retrieve the embedding
                ignore_unavailable=True,  # In case the index does not exist
            )
        except Exception as e:
            log.error(f"Error performing hybrid query on Elasticsearch index {self.collection_name}: {e}")
            raise

        return await self._results_to_chunks(results)

    async def delete(self):
        """Delete the entire Elasticsearch index with collection_name."""

        try:
            await self.client.indices.delete(index=self.collection_name, ignore_unavailable=True)
        except Exception as e:
            log.error(f"Error deleting Elasticsearch index {self.collection_name}: {e}")
            raise


class ElasticsearchVectorIOAdapter(OpenAIVectorStoreMixin, VectorIO, VectorStoresProtocolPrivate):
    """Vector I/O adapter for remote Elasticsearch instances."""

    def __init__(
        self,
        config: ElasticsearchVectorIOConfig,
        inference_api: Inference,
        files_api: Files | None = None,
        file_processor_api: FileProcessors | None = None,
    ) -> None:
        super().__init__(
            inference_api=inference_api, files_api=files_api, kvstore=None, file_processor_api=file_processor_api
        )
        self.config = config
        self.client: AsyncElasticsearch = None
        self.cache = {}
        self.vector_store_table = None
        self.metadata_collection_name = "openai_vector_stores_metadata"

    async def initialize(self) -> None:
        self.client = AsyncElasticsearch(hosts=self.config.elasticsearch_url, api_key=self.config.elasticsearch_api_key)
        self.kvstore = await kvstore_impl(self.config.persistence)

        start_key = VECTOR_DBS_PREFIX
        end_key = f"{VECTOR_DBS_PREFIX}\xff"
        stored_vector_stores = await self.kvstore.values_in_range(start_key, end_key)

        for vector_store_data in stored_vector_stores:
            vector_store = VectorStore.model_validate_json(vector_store_data)
            index = VectorStoreWithIndex(
                vector_store, ElasticsearchIndex(self.client, vector_store.identifier), self.inference_api
            )
            self.cache[vector_store.identifier] = index
        self.openai_vector_stores = await self._load_openai_vector_stores()

    async def shutdown(self) -> None:
        await self.client.close()
        # Clean up mixin resources (file batch tasks)
        await super().shutdown()

    async def register_vector_store(self, vector_store: VectorStore) -> None:
        assert self.kvstore is not None
        key = f"{VECTOR_DBS_PREFIX}{vector_store.identifier}"
        await self.kvstore.set(key=key, value=vector_store.model_dump_json())

        index = VectorStoreWithIndex(
            vector_store=vector_store,
            index=ElasticsearchIndex(self.client, vector_store.identifier),
            inference_api=self.inference_api,
        )

        self.cache[vector_store.identifier] = index

    async def unregister_vector_store(self, vector_store_id: str) -> None:
        if vector_store_id in self.cache:
            await self.cache[vector_store_id].index.delete()
            del self.cache[vector_store_id]

        assert self.kvstore is not None
        await self.kvstore.delete(f"{VECTOR_DBS_PREFIX}{vector_store_id}")

    async def _get_and_cache_vector_store_index(self, vector_store_id: str) -> VectorStoreWithIndex | None:
        if vector_store_id in self.cache:
            return self.cache[vector_store_id]

        if self.vector_store_table is None:
            raise ValueError(f"Vector DB not found {vector_store_id}")

        vector_store = await self.vector_store_table.get_vector_store(vector_store_id)
        if not vector_store:
            raise VectorStoreNotFoundError(vector_store_id)

        index = VectorStoreWithIndex(
            vector_store=vector_store,
            index=ElasticsearchIndex(client=self.client, collection_name=vector_store.identifier),
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
        """Delete chunks from an Elasticsearch vector store."""
        index = await self._get_and_cache_vector_store_index(request.vector_store_id)
        if not index:
            raise ValueError(f"Vector DB {request.vector_store_id} not found")

        await index.index.delete_chunks(request.chunks)
