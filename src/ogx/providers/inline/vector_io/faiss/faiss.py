# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import asyncio
import base64
import io
import json
import threading
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from numpy.typing import NDArray

_faiss: Any = None
_faiss_lock = threading.Lock()


def _get_faiss() -> Any:
    global _faiss
    if _faiss is not None:
        return _faiss
    with _faiss_lock:
        if _faiss is not None:
            return _faiss
        import faiss  # type: ignore[import-untyped]

        _faiss = faiss
        return _faiss


_numpy: Any = None
_numpy_lock = threading.Lock()


def _get_numpy() -> Any:
    global _numpy
    if _numpy is not None:
        return _numpy
    with _numpy_lock:
        if _numpy is not None:
            return _numpy
        import numpy

        _numpy = numpy
        return _numpy


from ogx.core.storage.kvstore import kvstore_impl
from ogx.log import get_logger
from ogx.providers.utils.memory.openai_vector_store_mixin import OpenAIVectorStoreMixin
from ogx.providers.utils.memory.vector_store import ChunkForDeletion, EmbeddingIndex, VectorStoreWithIndex
from ogx.providers.utils.vector_io import load_embedded_chunk_with_backward_compat
from ogx.providers.utils.vector_io.filters import CompoundFilter, Filter
from ogx_api import (
    DeleteChunksRequest,
    EmbeddedChunk,
    FileProcessors,
    Files,
    HealthResponse,
    HealthStatus,
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

from .config import FaissVectorIOConfig

logger = get_logger(name=__name__, category="vector_io")


def _list_op(mv: Any, fv: Any, *, negate: bool) -> bool:
    op = "nin" if negate else "in"
    if not isinstance(fv, list):
        raise ValueError(f"'{op}' filter requires a list value, got {type(fv)}")
    return (mv not in fv) if negate else (mv in fv)


_COMPARISON_OPS: dict[str, Any] = {
    "eq": lambda mv, fv: mv == fv,
    "ne": lambda mv, fv: mv != fv,
    "gt": lambda mv, fv: mv > fv,
    "gte": lambda mv, fv: mv >= fv,
    "lt": lambda mv, fv: mv < fv,
    "lte": lambda mv, fv: mv <= fv,
    "in": lambda mv, fv: _list_op(mv, fv, negate=False),
    "nin": lambda mv, fv: _list_op(mv, fv, negate=True),
}

VERSION = "v3"
VECTOR_DBS_PREFIX = f"vector_stores:{VERSION}::"
FAISS_INDEX_PREFIX = f"faiss_index:{VERSION}::"
OPENAI_VECTOR_STORES_PREFIX = f"openai_vector_stores:{VERSION}::"
OPENAI_VECTOR_STORES_FILES_PREFIX = f"openai_vector_stores_files:{VERSION}::"
OPENAI_VECTOR_STORES_FILES_CONTENTS_PREFIX = f"openai_vector_stores_files_contents:{VERSION}::"


class FaissIndex(EmbeddingIndex):
    """FAISS-based embedding index with optional KV store persistence."""

    def __init__(self, dimension: int, kvstore: KVStore | None = None, bank_id: str | None = None):
        self.index = _get_faiss().IndexFlatL2(dimension)
        self.chunk_by_index: dict[int, EmbeddedChunk] = {}
        self.kvstore = kvstore
        self.bank_id = bank_id

        # A list of chunk id's in the same order as they are in the index,
        # must be updated when chunks are added or removed
        self.chunk_id_lock = asyncio.Lock()
        self.chunk_ids: list[Any] = []

        # Inverted index: metadata_key -> metadata_value -> set of faiss positions
        # Rebuilt from chunk_by_index on initialize(); not persisted separately.
        self._meta_index: dict[str, dict[Any, set[int]]] = {}

    @classmethod
    async def create(cls, dimension: int, kvstore: KVStore | None = None, bank_id: str | None = None):
        instance = cls(dimension, kvstore, bank_id)
        await instance.initialize()
        return instance

    async def initialize(self) -> None:
        if not self.kvstore:
            return

        index_key = f"{FAISS_INDEX_PREFIX}{self.bank_id}"
        stored_data = await self.kvstore.get(index_key)

        if stored_data:
            data = json.loads(stored_data)
            self.chunk_by_index = {}
            for k, v in data["chunk_by_index"].items():
                chunk_data = json.loads(v)
                # Use generic backward compatibility utility
                self.chunk_by_index[int(k)] = load_embedded_chunk_with_backward_compat(chunk_data)

            buffer = io.BytesIO(base64.b64decode(data["faiss_index"]))
            try:
                self.index = _get_faiss().deserialize_index(_get_numpy().load(buffer, allow_pickle=False))
                self.chunk_ids = [embedded_chunk.chunk_id for embedded_chunk in self.chunk_by_index.values()]
                # Rebuild inverted metadata index from loaded chunks
                for pos, chunk in self.chunk_by_index.items():
                    for key, val in chunk.metadata.items():
                        self._meta_index.setdefault(key, {}).setdefault(val, set()).add(pos)
            except Exception as e:
                logger.debug("Failed to deserialize Faiss index", error=str(e), exc_info=True)
                raise ValueError(
                    "Error deserializing Faiss index from storage. If you recently upgraded your OGX, Faiss, "
                    "or NumPy versions, you may need to delete the index and re-create it again or downgrade versions.\n"
                    f"The problematic index is stored in the key value store {self.kvstore} under the key '{index_key}'."
                ) from e

    async def _save_index(self):
        if not self.kvstore or not self.bank_id:
            return

        np_index = _get_faiss().serialize_index(self.index)
        buffer = io.BytesIO()
        _get_numpy().save(buffer, np_index, allow_pickle=False)
        data = {
            "chunk_by_index": {k: v.model_dump_json() for k, v in self.chunk_by_index.items()},
            "faiss_index": base64.b64encode(buffer.getvalue()).decode("utf-8"),
        }

        index_key = f"{FAISS_INDEX_PREFIX}{self.bank_id}"
        await self.kvstore.set(key=index_key, value=json.dumps(data))

    async def delete(self):
        if not self.kvstore or not self.bank_id:
            return

        await self.kvstore.delete(f"{FAISS_INDEX_PREFIX}{self.bank_id}")

    async def add_chunks(self, embedded_chunks: list[EmbeddedChunk]):
        if not embedded_chunks:
            return

        # Extract embeddings and validate dimensions
        np = _get_numpy()
        embeddings = np.array([ec.embedding for ec in embedded_chunks], dtype=np.float32)
        embedding_dim = embeddings.shape[1] if len(embeddings.shape) > 1 else embeddings.shape[0]
        if embedding_dim != self.index.d:
            raise ValueError(f"Embedding dimension mismatch. Expected {self.index.d}, got {embedding_dim}")

        # Store chunks by index and update inverted metadata index
        indexlen = len(self.chunk_by_index)
        for i, embedded_chunk in enumerate(embedded_chunks):
            faiss_pos = indexlen + i
            self.chunk_by_index[faiss_pos] = embedded_chunk
            for key, val in embedded_chunk.metadata.items():
                self._meta_index.setdefault(key, {}).setdefault(val, set()).add(faiss_pos)

        async with self.chunk_id_lock:
            self.index.add(embeddings)
            self.chunk_ids.extend([ec.chunk_id for ec in embedded_chunks])  # EmbeddedChunk inherits from Chunk

        # Save updated index
        await self._save_index()

    async def delete_chunks(self, chunks_for_deletion: list[ChunkForDeletion]) -> None:
        chunk_ids = [c.chunk_id for c in chunks_for_deletion]
        if not set(chunk_ids).issubset(self.chunk_ids):
            return

        def remove_chunk(chunk_id: str):
            removed_pos = self.chunk_ids.index(chunk_id)
            self.index.remove_ids(_get_numpy().array([removed_pos]))

            # Remove deleted position from _meta_index
            for val_map in self._meta_index.values():
                for positions in val_map.values():
                    positions.discard(removed_pos)

            new_chunk_by_index = {}
            for idx, chunk in self.chunk_by_index.items():
                # Shift all chunks after the removed chunk to the left
                if idx > removed_pos:
                    new_chunk_by_index[idx - 1] = chunk
                else:
                    new_chunk_by_index[idx] = chunk
            self.chunk_by_index = new_chunk_by_index
            self.chunk_ids.pop(removed_pos)

            # Shift all _meta_index positions > removed_pos down by 1
            for val_map in self._meta_index.values():
                for val in list(val_map):
                    val_map[val] = {p - 1 if p > removed_pos else p for p in val_map[val]}
                    if not val_map[val]:
                        del val_map[val]

        async with self.chunk_id_lock:
            for chunk_id in chunk_ids:
                remove_chunk(chunk_id)

        await self._save_index()

    def _resolve_filter_positions(self, filter_obj: Filter) -> set[int]:
        """
        Return the set of faiss positions that match filter_obj.

        Uses the inverted _meta_index for O(1) equality/set lookups and falls
        back to a linear scan of chunk_by_index for range ops (gt/gte/lt/lte/ne).
        """
        if isinstance(filter_obj, CompoundFilter):
            sub_sets = [self._resolve_filter_positions(f) for f in filter_obj.filters]
            if not sub_sets:
                return set()
            if filter_obj.type == "and":
                return set.intersection(*sub_sets)
            else:  # "or"
                return set.union(*sub_sets)

        # ComparisonFilter
        key, value, op_type = filter_obj.key, filter_obj.value, filter_obj.type
        if op_type == "eq":
            return self._meta_index.get(key, {}).get(value, set()).copy()
        if op_type == "in":
            result: set[int] = set()
            for v in value:
                result |= self._meta_index.get(key, {}).get(v, set())
            return result
        if op_type == "nin":
            excluded: set[int] = set()
            for v in value:
                excluded |= self._meta_index.get(key, {}).get(v, set())
            all_positions = {pos for s in self._meta_index.get(key, {}).values() for pos in s}
            return all_positions - excluded
        # Range ops and ne: linear scan over chunk_by_index metadata
        op = _COMPARISON_OPS.get(op_type)
        if op is None:
            raise ValueError(f"Unknown comparison operator: {op_type}")
        return {
            pos
            for pos, chunk in self.chunk_by_index.items()
            if key in chunk.metadata and op(chunk.metadata[key], value)
        }

    async def query_vector(
        self, embedding: "NDArray", k: int, score_threshold: float, filters: Filter | None = None
    ) -> QueryChunksResponse:
        """
        Performs vector-based search using Faiss similarity search.
        When filters are provided, pre-computes matching positions via the
        inverted _meta_index and passes them directly to Faiss via IDSelectorBatch,
        eliminating the need for post-hoc filtering or over-retrieval heuristics.
        """
        np = _get_numpy()
        faiss = _get_faiss()
        if filters is not None:
            candidate_positions = self._resolve_filter_positions(filters)
            if not candidate_positions:
                return QueryChunksResponse(chunks=[], scores=[])
            sel = faiss.IDSelectorBatch(np.array(sorted(candidate_positions), dtype=np.int64))
            params = faiss.SearchParameters(sel=sel)
            distances, indices = await asyncio.to_thread(
                lambda: self.index.search(embedding.reshape(1, -1).astype(np.float32), k, params=params)
            )
        else:
            distances, indices = await asyncio.to_thread(
                self.index.search, embedding.reshape(1, -1).astype(np.float32), k
            )

        chunks: list[EmbeddedChunk] = []
        scores: list[float] = []
        for d, i in zip(distances[0], indices[0], strict=False):
            if i < 0:
                continue
            score = 1.0 / float(d) if d != 0 else float("inf")
            if score < score_threshold:
                continue
            chunks.append(self.chunk_by_index[int(i)])
            scores.append(score)

        return QueryChunksResponse(chunks=chunks, scores=scores)

    async def query_keyword(
        self, query_string: str, k: int, score_threshold: float, filters: Filter | None = None
    ) -> QueryChunksResponse:
        raise NotImplementedError(
            "Keyword search is not supported - underlying DB FAISS does not support this search mode"
        )

    async def query_hybrid(
        self,
        embedding: "NDArray",
        query_string: str,
        k: int,
        score_threshold: float,
        reranker_type: str,
        reranker_params: dict[str, Any] | None = None,
        filters: Filter | None = None,
    ) -> QueryChunksResponse:
        raise NotImplementedError(
            "Hybrid search is not supported - underlying DB FAISS does not support this search mode"
        )


class FaissVectorIOAdapter(OpenAIVectorStoreMixin, VectorIO, VectorStoresProtocolPrivate):
    """VectorIO adapter that uses FAISS for similarity search and vector storage."""

    def __init__(
        self,
        config: FaissVectorIOConfig,
        inference_api: Inference,
        files_api: Files | None,
        file_processor_api: FileProcessors | None = None,
    ) -> None:
        super().__init__(
            inference_api=inference_api, files_api=files_api, kvstore=None, file_processor_api=file_processor_api
        )
        self.config = config
        self.cache: dict[str, VectorStoreWithIndex] = {}

    async def initialize(self) -> None:
        self.kvstore = await kvstore_impl(self.config.persistence)
        # Load existing banks from kvstore
        start_key = VECTOR_DBS_PREFIX
        end_key = f"{VECTOR_DBS_PREFIX}\xff"
        stored_vector_stores = await self.kvstore.values_in_range(start_key, end_key)

        for vector_store_data in stored_vector_stores:
            vector_store = VectorStore.model_validate_json(vector_store_data)
            index = VectorStoreWithIndex(
                vector_store,
                await FaissIndex.create(vector_store.embedding_dimension, self.kvstore, vector_store.identifier),
                self.inference_api,
            )
            self.cache[vector_store.identifier] = index

        # Load existing OpenAI vector stores into the in-memory cache
        await self.initialize_openai_vector_stores()

    async def shutdown(self) -> None:
        # Clean up mixin resources (file batch tasks)
        await super().shutdown()

    async def health(self) -> HealthResponse:
        """
        Performs a health check by verifying connectivity to the inline faiss DB.
        This method is used by the Provider API to verify
        that the service is running correctly.
        Returns:

            HealthResponse: A dictionary containing the health status.
        """
        try:
            vector_dimension = 128  # sample dimension
            _get_faiss().IndexFlatL2(vector_dimension)
            return HealthResponse(status=HealthStatus.OK)
        except Exception as e:
            return HealthResponse(status=HealthStatus.ERROR, message=f"Health check failed: {str(e)}")

    async def register_vector_store(self, vector_store: VectorStore) -> None:
        if self.kvstore is None:
            raise RuntimeError("KVStore not initialized. Call initialize() before registering vector stores.")

        key = f"{VECTOR_DBS_PREFIX}{vector_store.identifier}"
        await self.kvstore.set(key=key, value=vector_store.model_dump_json())

        # Store in cache
        self.cache[vector_store.identifier] = VectorStoreWithIndex(
            vector_store=vector_store,
            index=await FaissIndex.create(vector_store.embedding_dimension, self.kvstore, vector_store.identifier),
            inference_api=self.inference_api,
        )

    async def list_vector_stores(self) -> list[VectorStore]:
        return [i.vector_store for i in self.cache.values()]

    async def unregister_vector_store(self, vector_store_id: str) -> None:
        if self.kvstore is None:
            raise RuntimeError("KVStore not initialized. Call initialize() before unregistering vector stores.")

        if vector_store_id not in self.cache:
            return

        await self.cache[vector_store_id].index.delete()
        del self.cache[vector_store_id]
        await self.kvstore.delete(f"{VECTOR_DBS_PREFIX}{vector_store_id}")

    async def _get_and_cache_vector_store_index(self, vector_store_id: str) -> VectorStoreWithIndex | None:
        if vector_store_id in self.cache:
            return self.cache[vector_store_id]

        if self.kvstore is None:
            raise RuntimeError("KVStore not initialized. Call initialize() before using vector stores.")

        key = f"{VECTOR_DBS_PREFIX}{vector_store_id}"
        vector_store_data = await self.kvstore.get(key)
        if not vector_store_data:
            raise VectorStoreNotFoundError(vector_store_id)

        vector_store = VectorStore.model_validate_json(vector_store_data)
        index = VectorStoreWithIndex(
            vector_store=vector_store,
            index=await FaissIndex.create(vector_store.embedding_dimension, self.kvstore, vector_store.identifier),
            inference_api=self.inference_api,
        )
        self.cache[vector_store_id] = index
        return index

    async def insert_chunks(self, request: InsertChunksRequest) -> None:
        index = self.cache.get(request.vector_store_id)
        if index is None:
            raise VectorStoreNotFoundError(request.vector_store_id)

        await index.insert_chunks(request)

    async def query_chunks(self, request: QueryChunksRequest) -> QueryChunksResponse:
        index = self.cache.get(request.vector_store_id)
        if index is None:
            raise VectorStoreNotFoundError(request.vector_store_id)

        return await index.query_chunks(request)

    async def delete_chunks(self, request: DeleteChunksRequest) -> None:
        """Delete chunks from a faiss index"""
        faiss_index = self.cache[request.vector_store_id].index
        await faiss_index.delete_chunks(request.chunks)
