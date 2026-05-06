# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import asyncio
import json
import re
import sqlite3
import struct
import threading
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from numpy.typing import NDArray

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


_sqlite_vec: Any = None
_sqlite_vec_lock = threading.Lock()


def _get_sqlite_vec() -> Any:
    global _sqlite_vec
    if _sqlite_vec is not None:
        return _sqlite_vec
    with _sqlite_vec_lock:
        if _sqlite_vec is not None:
            return _sqlite_vec
        import sqlite_vec  # type: ignore[import-untyped]

        _sqlite_vec = sqlite_vec
        return _sqlite_vec


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
from ogx.providers.utils.vector_io.filters import ComparisonFilter, CompoundFilter, Filter
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
    VectorStoreNotFoundError,
    VectorStoresProtocolPrivate,
)
from ogx_api.internal.kvstore import KVStore

logger = get_logger(name=__name__, category="vector_io")

_SQL_OPS: dict[str, str] = {
    "eq": "=",
    "ne": "!=",
    "gt": ">",
    "gte": ">=",
    "lt": "<",
    "lte": "<=",
}

# Specifying search mode is dependent on the VectorIO provider.
VECTOR_SEARCH = "vector"
KEYWORD_SEARCH = "keyword"
HYBRID_SEARCH = "hybrid"
SEARCH_MODES = {VECTOR_SEARCH, KEYWORD_SEARCH, HYBRID_SEARCH}

VERSION = "v3"
VECTOR_DBS_PREFIX = f"vector_stores:sqlite_vec:{VERSION}::"
VECTOR_INDEX_PREFIX = f"vector_index:sqlite_vec:{VERSION}::"
OPENAI_VECTOR_STORES_PREFIX = f"openai_vector_stores:sqlite_vec:{VERSION}::"
OPENAI_VECTOR_STORES_FILES_PREFIX = f"openai_vector_stores_files:sqlite_vec:{VERSION}::"
OPENAI_VECTOR_STORES_FILES_CONTENTS_PREFIX = f"openai_vector_stores_files_contents:sqlite_vec:{VERSION}::"


def serialize_vector(vector: list[float]) -> bytes:
    """Serialize a list of floats into a compact binary representation."""
    return struct.pack(f"{len(vector)}f", *vector)


def _create_sqlite_connection(db_path: str):
    """Create a SQLite connection with sqlite_vec extension loaded."""
    connection = sqlite3.connect(db_path)
    connection.enable_load_extension(True)
    _get_sqlite_vec().load(connection)
    connection.enable_load_extension(False)
    return connection


def _make_sql_identifier(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]", "_", name)


class SQLiteVecIndex(EmbeddingIndex):
    """
    An index implementation that stores embeddings in a SQLite virtual table using sqlite-vec.
    Two tables are used:
      - A metadata table (chunks_{bank_id}) that holds the chunk JSON.
      - A virtual table (vec_chunks_{bank_id}) that holds the serialized vector.
      - An FTS5 table (fts_chunks_{bank_id}) for full-text keyword search.
    """

    def __init__(self, dimension: int, db_path: str, bank_id: str, kvstore: KVStore | None = None):
        self.dimension = dimension
        self.db_path = db_path
        self.bank_id = bank_id
        self.metadata_table = _make_sql_identifier(f"chunks_{bank_id}")
        self.vector_table = _make_sql_identifier(f"vec_chunks_{bank_id}")
        self.fts_table = _make_sql_identifier(f"fts_chunks_{bank_id}")
        self.kvstore = kvstore

    @classmethod
    async def create(cls, dimension: int, db_path: str, bank_id: str):
        instance = cls(dimension, db_path, bank_id)
        await instance.initialize()
        return instance

    async def initialize(self) -> None:
        def _init_tables():
            connection = _create_sqlite_connection(self.db_path)
            cur = connection.cursor()
            try:
                # Create the table to store chunk metadata.
                cur.execute(f"""
                    CREATE TABLE IF NOT EXISTS [{self.metadata_table}] (
                        id TEXT PRIMARY KEY,
                        chunk TEXT
                    );
                """)
                # Create the virtual table for embeddings.
                cur.execute(f"""
                    CREATE VIRTUAL TABLE IF NOT EXISTS [{self.vector_table}]
                    USING vec0(embedding FLOAT[{self.dimension}], id TEXT);
                """)
                connection.commit()
                # FTS5 table (for keyword search) - creating both the tables by default. Will use the relevant one
                # based on query. Implementation of the change on client side will allow passing the search_mode option
                # during initialization to make it easier to create the table that is required.
                cur.execute(f"""
                            CREATE VIRTUAL TABLE IF NOT EXISTS [{self.fts_table}]
                            USING fts5(id, content);
                        """)
                connection.commit()
            finally:
                cur.close()
                connection.close()

        await asyncio.to_thread(_init_tables)

    async def delete(self) -> None:
        def _drop_tables():
            connection = _create_sqlite_connection(self.db_path)
            cur = connection.cursor()
            try:
                cur.execute(f"DROP TABLE IF EXISTS [{self.metadata_table}];")
                cur.execute(f"DROP TABLE IF EXISTS [{self.vector_table}];")
                cur.execute(f"DROP TABLE IF EXISTS [{self.fts_table}];")
                connection.commit()
            finally:
                cur.close()
                connection.close()

        await asyncio.to_thread(_drop_tables)

    async def add_chunks(self, embedded_chunks: list[EmbeddedChunk], batch_size: int = 500):
        """
        Add new embedded chunks using batch inserts.
        For each embedded chunk, we insert the chunk JSON into the metadata table and then insert its
        embedding (serialized to raw bytes) into the virtual table using the assigned rowid.
        If any insert fails, the transaction is rolled back to maintain consistency.
        Also inserts chunk content into FTS table for keyword search support.
        """
        chunks = embedded_chunks  # EmbeddedChunk now inherits from Chunk
        np = _get_numpy()
        embeddings = np.array([ec.embedding for ec in embedded_chunks], dtype=np.float32)
        assert all(isinstance(chunk.content, str) for chunk in chunks), "SQLiteVecIndex only supports text chunks"

        def _execute_all_batch_inserts():
            connection = _create_sqlite_connection(self.db_path)
            cur = connection.cursor()

            try:
                cur.execute("BEGIN TRANSACTION")
                for i in range(0, len(chunks), batch_size):
                    batch_chunks = chunks[i : i + batch_size]
                    batch_embeddings = embeddings[i : i + batch_size]

                    # Insert metadata
                    metadata_data = [(chunk.chunk_id, chunk.model_dump_json()) for chunk in batch_chunks]
                    cur.executemany(
                        f"""
                        INSERT INTO [{self.metadata_table}] (id, chunk)
                        VALUES (?, ?)
                        ON CONFLICT(id) DO UPDATE SET chunk = excluded.chunk;
                        """,
                        metadata_data,
                    )

                    # Insert vector embeddings
                    embedding_data = [
                        ((chunk.chunk_id, serialize_vector(emb.tolist())))
                        for chunk, emb in zip(batch_chunks, batch_embeddings, strict=True)
                    ]
                    cur.executemany(f"INSERT INTO [{self.vector_table}] (id, embedding) VALUES (?, ?);", embedding_data)

                    # Insert FTS content
                    fts_data = [(chunk.chunk_id, chunk.content) for chunk in batch_chunks]
                    # DELETE existing entries with same IDs (FTS5 doesn't support ON CONFLICT)
                    cur.executemany(f"DELETE FROM [{self.fts_table}] WHERE id = ?;", [(row[0],) for row in fts_data])

                    # INSERT new entries
                    cur.executemany(f"INSERT INTO [{self.fts_table}] (id, content) VALUES (?, ?);", fts_data)

                connection.commit()

            except sqlite3.Error as e:
                connection.rollback()
                logger.error("Error inserting into", vector_table=self.vector_table, error=str(e))
                raise

            finally:
                cur.close()
                connection.close()

        # Run batch insertion in a background thread
        await asyncio.to_thread(_execute_all_batch_inserts)

    def _translate_filters(self, filters: Filter | None) -> tuple[str, list[Any]]:
        """Translate OpenAI-compatible filters to SQL WHERE clause with parameters.

        Args:
            filters: The filter to translate (ComparisonFilter or CompoundFilter)

        Returns:
            A tuple of (where_clause, parameters) where where_clause is a SQL condition string
            and parameters is a list of values to be bound to the query.
        """
        if filters is None:
            return "", []

        return self._translate_single_filter(filters)

    def _translate_single_filter(self, filter_obj: Filter) -> tuple[str, list[Any]]:
        """Translate a single filter to SQL."""
        if isinstance(filter_obj, ComparisonFilter):
            return self._translate_comparison_filter(filter_obj)
        elif isinstance(filter_obj, CompoundFilter):
            return self._translate_compound_filter(filter_obj)
        else:
            raise ValueError(f"Unknown filter type: {type(filter_obj)}")

    def _translate_comparison_filter(self, filter_obj: ComparisonFilter) -> tuple[str, list[Any]]:
        """Translate a comparison filter to SQL WHERE clause."""
        key, value, op_type = filter_obj.key, filter_obj.value, filter_obj.type
        json_path = f"$.metadata.{key}"
        expr = f"JSON_EXTRACT(m.chunk, '{json_path}')"

        if op_type in _SQL_OPS:
            return f"{expr} {_SQL_OPS[op_type]} ?", [value]
        elif op_type == "in":
            if not isinstance(value, list):
                raise ValueError(f"'in' filter requires a list value, got {type(value)}")
            placeholders = ", ".join("?" * len(value))
            return f"{expr} IN ({placeholders})", value
        elif op_type == "nin":
            if not isinstance(value, list):
                raise ValueError(f"'nin' filter requires a list value, got {type(value)}")
            placeholders = ", ".join("?" * len(value))
            return f"{expr} NOT IN ({placeholders})", value
        else:
            raise ValueError(f"Unknown comparison operator: {op_type}")

    def _translate_compound_filter(self, filter_obj: CompoundFilter) -> tuple[str, list[Any]]:
        """Translate a compound filter (and/or) to SQL WHERE clause."""
        if not filter_obj.filters:
            return "", []

        clauses = []
        params: list[Any] = []

        for sub_filter in filter_obj.filters:
            clause, sub_params = self._translate_single_filter(sub_filter)
            if clause:
                clauses.append(f"({clause})")
                params.extend(sub_params)

        if not clauses:
            return "", []

        operator = " AND " if filter_obj.type == "and" else " OR "
        return operator.join(clauses), params

    async def query_vector(
        self, embedding: "NDArray", k: int, score_threshold: float, filters: Filter | None = None
    ) -> QueryChunksResponse:
        """
        Performs vector-based search using a virtual table for vector similarity.
        Optionally filters results based on metadata using SQL WHERE clauses.
        """
        # Translate filters to SQL WHERE clause
        filter_clause, filter_params = self._translate_filters(filters)

        def _execute_query():
            connection = _create_sqlite_connection(self.db_path)
            cur = connection.cursor()
            try:
                emb_list = embedding.tolist() if isinstance(embedding, _get_numpy().ndarray) else list(embedding)
                emb_blob = serialize_vector(emb_list)

                # Build query with optional filter clause
                query_sql = f"""
                    SELECT m.id, m.chunk, v.distance
                    FROM [{self.vector_table}] AS v
                    JOIN [{self.metadata_table}] AS m ON m.id = v.id
                    WHERE v.embedding MATCH ? AND k = ?
                    {"AND (" + filter_clause + ")" if filter_clause else ""}
                    ORDER BY v.distance;
                """
                cur.execute(query_sql, (emb_blob, k, *filter_params))
                return cur.fetchall()
            finally:
                cur.close()
                connection.close()

        rows = await asyncio.to_thread(_execute_query)
        chunks, scores = [], []
        for row in rows:
            _id, chunk_json, distance = row
            score = 1.0 / distance if distance != 0 else float("inf")
            if score < score_threshold:
                continue
            try:
                chunk_data = json.loads(chunk_json)
                embedded_chunk = load_embedded_chunk_with_backward_compat(chunk_data)
            except Exception as e:
                logger.error("Error parsing chunk JSON for id", _id=_id, error=str(e))
                continue
            chunks.append(embedded_chunk)
            scores.append(score)
        return QueryChunksResponse(chunks=chunks, scores=scores)

    async def query_keyword(
        self, query_string: str, k: int, score_threshold: float, filters: Filter | None = None
    ) -> QueryChunksResponse:
        """
        Performs keyword-based search using SQLite FTS5 for relevance-ranked full-text search.
        Optionally filters results based on metadata using SQL WHERE clauses.
        """
        # Translate filters to SQL WHERE clause
        filter_clause, filter_params = self._translate_filters(filters)

        def _execute_query():
            connection = _create_sqlite_connection(self.db_path)
            cur = connection.cursor()
            try:
                # Build query with optional filter clause
                query_sql = f"""
                    SELECT DISTINCT m.id, m.chunk, bm25([{self.fts_table}]) AS score
                    FROM [{self.fts_table}] AS f
                    JOIN [{self.metadata_table}] AS m ON m.id = f.id
                    WHERE f.content MATCH ?
                    {"AND (" + filter_clause + ")" if filter_clause else ""}
                    ORDER BY score ASC
                    LIMIT ?;
                """
                cur.execute(query_sql, (query_string, *filter_params, k))
                return cur.fetchall()
            finally:
                cur.close()
                connection.close()

        rows = await asyncio.to_thread(_execute_query)
        chunks, scores = [], []
        for row in rows:
            _id, chunk_json, score = row
            # BM25 scores returned by sqlite-vec are NEGATED (i.e., more relevant = more negative).
            # This design is intentional to simplify sorting by ascending score.
            # Reference: https://alexgarcia.xyz/blog/2024/sqlite-vec-hybrid-search/index.html
            if score > -score_threshold:
                continue
            try:
                chunk_data = json.loads(chunk_json)
                embedded_chunk = load_embedded_chunk_with_backward_compat(chunk_data)
            except Exception as e:
                logger.error("Error parsing chunk JSON for id", _id=_id, error=str(e))
                continue
            chunks.append(embedded_chunk)
            # Negate so higher = more relevant, matching the convention
            # expected by RRF and other downstream rerankers.
            scores.append(-score)
        return QueryChunksResponse(chunks=chunks, scores=scores)

    async def query_hybrid(
        self,
        embedding: "NDArray",
        query_string: str,
        k: int,
        score_threshold: float,
        reranker_type: str = RERANKER_TYPE_RRF,
        reranker_params: dict[str, Any] | None = None,
        filters: Filter | None = None,
    ) -> QueryChunksResponse:
        """
        Hybrid search using a configurable re-ranking strategy.
        Optionally filters results based on metadata using SQL WHERE clauses.

        Args:
            embedding: The query embedding vector
            query_string: The text query for keyword search
            k: Number of results to return
            score_threshold: Minimum similarity score threshold
            reranker_type: Type of reranker to use ("rrf" or "weighted")
            reranker_params: Parameters for the reranker
            filters: Optional metadata filters to apply to results

        Returns:
            QueryChunksResponse with combined results
        """
        if reranker_params is None:
            reranker_params = {}

        # Get results from both search methods, passing filters to each
        vector_response = await self.query_vector(embedding, k, score_threshold, filters)
        keyword_response = await self.query_keyword(query_string, k, score_threshold, filters)

        # Convert responses to score dictionaries using chunk_id (EmbeddedChunk inherits from Chunk)
        vector_scores = {
            embedded_chunk.chunk_id: score
            for embedded_chunk, score in zip(vector_response.chunks, vector_response.scores, strict=False)
        }
        keyword_scores = {
            embedded_chunk.chunk_id: score
            for embedded_chunk, score in zip(keyword_response.chunks, keyword_response.scores, strict=False)
        }

        # Combine scores using the reranking utility
        combined_scores = WeightedInMemoryAggregator.combine_search_results(
            vector_scores, keyword_scores, reranker_type, reranker_params
        )

        # Sort by combined score and get top k results
        sorted_items = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        top_k_items = sorted_items[:k]

        # Filter by score threshold
        filtered_items = [(doc_id, score) for doc_id, score in top_k_items if score >= score_threshold]

        # Create a map of chunk_id to embedded_chunk for both responses
        chunk_map = {ec.chunk_id: ec for ec in vector_response.chunks + keyword_response.chunks}

        # Use the map to look up embedded chunks by their IDs
        chunks = []
        scores = []
        for doc_id, score in filtered_items:
            if doc_id in chunk_map:
                chunks.append(chunk_map[doc_id])
                scores.append(score)

        return QueryChunksResponse(chunks=chunks, scores=scores)

    async def delete_chunks(self, chunks_for_deletion: list[ChunkForDeletion]) -> None:
        """Remove a chunk from the SQLite vector store."""
        chunk_ids = [c.chunk_id for c in chunks_for_deletion]

        def _delete_chunks():
            connection = _create_sqlite_connection(self.db_path)
            cur = connection.cursor()
            try:
                cur.execute("BEGIN TRANSACTION")

                # Delete from metadata table
                placeholders = ",".join("?" * len(chunk_ids))
                cur.execute(f"DELETE FROM {self.metadata_table} WHERE id IN ({placeholders})", chunk_ids)

                # Delete from vector table
                cur.execute(f"DELETE FROM {self.vector_table} WHERE id IN ({placeholders})", chunk_ids)

                # Delete from FTS table
                cur.execute(f"DELETE FROM {self.fts_table} WHERE id IN ({placeholders})", chunk_ids)

                connection.commit()
            except Exception as e:
                connection.rollback()
                logger.error("Error deleting chunks", error=str(e))
                raise
            finally:
                cur.close()
                connection.close()

        await asyncio.to_thread(_delete_chunks)


class SQLiteVecVectorIOAdapter(OpenAIVectorStoreMixin, VectorIO, VectorStoresProtocolPrivate):
    """
    A VectorIO implementation using SQLite + sqlite_vec.
    This class handles vector database registration (with metadata stored in a table named `vector_stores`)
    and creates a cache of VectorStoreWithIndex instances (each wrapping a SQLiteVecIndex).
    """

    def __init__(
        self,
        config,
        inference_api: Inference,
        files_api: Files | None,
        file_processor_api: FileProcessors | None = None,
    ) -> None:
        super().__init__(
            inference_api=inference_api, files_api=files_api, kvstore=None, file_processor_api=file_processor_api
        )
        self.config = config
        self.cache: dict[str, VectorStoreWithIndex] = {}
        self.vector_store_table = None

    async def initialize(self) -> None:
        self.kvstore = await kvstore_impl(self.config.persistence)

        start_key = VECTOR_DBS_PREFIX
        end_key = f"{VECTOR_DBS_PREFIX}\xff"
        stored_vector_stores = await self.kvstore.values_in_range(start_key, end_key)
        for db_json in stored_vector_stores:
            vector_store = VectorStore.model_validate_json(db_json)
            index = await SQLiteVecIndex.create(
                vector_store.embedding_dimension, self.config.db_path, vector_store.identifier
            )
            self.cache[vector_store.identifier] = VectorStoreWithIndex(vector_store, index, self.inference_api)

        # Load existing OpenAI vector stores into the in-memory cache
        await self.initialize_openai_vector_stores()

    async def shutdown(self) -> None:
        # Clean up mixin resources (file batch tasks)
        await super().shutdown()

    async def list_vector_stores(self) -> list[VectorStore]:
        return [v.vector_store for v in self.cache.values()]

    async def register_vector_store(self, vector_store: VectorStore) -> None:
        if self.kvstore is None:
            raise RuntimeError("KVStore not initialized. Call initialize() before registering vector stores.")

        # Save to kvstore for persistence
        key = f"{VECTOR_DBS_PREFIX}{vector_store.identifier}"
        await self.kvstore.set(key=key, value=vector_store.model_dump_json())

        # Create and cache the index
        index = await SQLiteVecIndex.create(
            vector_store.embedding_dimension, self.config.db_path, vector_store.identifier
        )
        self.cache[vector_store.identifier] = VectorStoreWithIndex(vector_store, index, self.inference_api)

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
        index = VectorStoreWithIndex(
            vector_store=vector_store,
            index=SQLiteVecIndex(
                dimension=vector_store.embedding_dimension,
                db_path=self.config.db_path,
                bank_id=vector_store.identifier,
                kvstore=self.kvstore,
            ),
            inference_api=self.inference_api,
        )
        self.cache[vector_store_id] = index
        return index

    async def unregister_vector_store(self, vector_store_id: str) -> None:
        if vector_store_id not in self.cache:
            return
        await self.cache[vector_store_id].index.delete()
        del self.cache[vector_store_id]

    async def insert_chunks(self, request: InsertChunksRequest) -> None:
        index = await self._get_and_cache_vector_store_index(request.vector_store_id)
        if not index:
            raise VectorStoreNotFoundError(request.vector_store_id)
        # The VectorStoreWithIndex helper validates embeddings and calls the index's add_chunks method
        await index.insert_chunks(request)

    async def query_chunks(self, request: QueryChunksRequest) -> QueryChunksResponse:
        index = await self._get_and_cache_vector_store_index(request.vector_store_id)
        if not index:
            raise VectorStoreNotFoundError(request.vector_store_id)

        return await index.query_chunks(request)

    async def delete_chunks(self, request: DeleteChunksRequest) -> None:
        """Delete chunks from a sqlite_vec index."""
        index = await self._get_and_cache_vector_store_index(request.vector_store_id)
        if not index:
            raise VectorStoreNotFoundError(request.vector_store_id)

        await index.index.delete_chunks(request.chunks)
