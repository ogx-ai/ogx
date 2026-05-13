# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import asyncio
import heapq
import json
import re
from typing import Any

import asyncpg
from numpy.typing import NDArray
from pgvector.asyncpg import register_vector
from pydantic import BaseModel

from ogx.core.storage.kvstore import kvstore_impl
from ogx.log import get_logger
from ogx.providers.utils.inference.prompt_adapter import interleaved_content_as_str
from ogx.providers.utils.memory.openai_vector_store_mixin import OpenAIVectorStoreMixin
from ogx.providers.utils.memory.vector_store import ChunkForDeletion, EmbeddingIndex, VectorStoreWithIndex
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

from .config import PGVectorIndexConfig, PGVectorIndexType, PGVectorVectorIOConfig

log = get_logger(name=__name__, category="vector_io::pgvector")

VERSION = "v3"
VECTOR_DBS_PREFIX = f"vector_stores:pgvector:{VERSION}::"
VECTOR_INDEX_PREFIX = f"vector_index:pgvector:{VERSION}::"
OPENAI_VECTOR_STORES_PREFIX = f"openai_vector_stores:pgvector:{VERSION}::"
OPENAI_VECTOR_STORES_FILES_PREFIX = f"openai_vector_stores_files:pgvector:{VERSION}::"
OPENAI_VECTOR_STORES_FILES_CONTENTS_PREFIX = f"openai_vector_stores_files_contents:pgvector:{VERSION}::"

_PG_SQL_OPS: dict[str, str] = {
    "eq": "=",
    "ne": "!=",
    "gt": ">",
    "gte": ">=",
    "lt": "<",
    "lte": "<=",
}

# Regex for validating metadata key names to prevent SQL injection in JSONB paths
_VALID_KEY_PATTERN = re.compile(r"^[a-zA-Z0-9_\-]+$")


def _quote_ident(name: str) -> str:
    """Quote a PostgreSQL identifier (table/index name) per SQL standard."""
    return '"' + name.replace('"', '""') + '"'


async def check_extension_version(conn: asyncpg.Connection) -> str | None:
    """Query the installed pgvector extension version.

    Args:
        conn: asyncpg connection

    Returns:
        Version string if the extension is installed, otherwise None
    """
    return await conn.fetchval("SELECT extversion FROM pg_extension WHERE extname = 'vector'")


async def create_vector_extension(conn: asyncpg.Connection) -> None:
    """Create the pgvector extension in the database.

    Args:
        conn: asyncpg connection
    """
    try:
        log.info("Vector extension not found, creating...")
        await conn.execute("CREATE EXTENSION vector;")
        log.info("Vector extension created successfully")
        version = await check_extension_version(conn)
        log.info("Vector extension version", version=version)

    except asyncpg.PostgresError as e:
        raise RuntimeError(f"Failed to create vector extension for PGVector: {e}") from e


async def upsert_models(pool: asyncpg.Pool, keys_models: list[tuple[str, BaseModel]]) -> None:
    """Insert or update serialized Pydantic models in the metadata_store table.

    Args:
        pool: asyncpg connection pool
        keys_models: list of (key, model) tuples to upsert
    """
    async with pool.acquire() as conn:
        values = [(key, json.dumps(model.model_dump())) for key, model in keys_models]
        await conn.executemany(
            """
            INSERT INTO metadata_store (key, data)
            VALUES ($1, $2::jsonb)
            ON CONFLICT (key) DO UPDATE
            SET data = EXCLUDED.data
            """,
            values,
        )


async def remove_vector_store_metadata(pool: asyncpg.Pool, vector_store_id: str) -> None:
    """Performs removal of vector store metadata from PGVector metadata_store table when vector store is unregistered.

    Args:
        pool: asyncpg connection pool
        vector_store_id: identifier of VectorStore resource
    """
    try:
        async with pool.acquire() as conn:
            result = await conn.execute("DELETE FROM metadata_store WHERE key = $1", vector_store_id)
            if result and result != "DELETE 0":
                log.info(
                    "Removed metadata for vector store from PGVector metadata_store table",
                    vector_store_id=vector_store_id,
                )

    except Exception as e:
        raise RuntimeError(
            f"Error removing metadata from PGVector metadata_store for vector_store: {vector_store_id}"
        ) from e


class PGVectorIndex(EmbeddingIndex):
    """Embedding index backed by PostgreSQL with the pgvector extension."""

    # reference: https://github.com/pgvector/pgvector?tab=readme-ov-file#querying
    # OGX supports only search functions that are applied for embeddings with vector type
    PGVECTOR_DISTANCE_METRIC_TO_SEARCH_FUNCTION: dict[str, str] = {
        "L2": "<->",
        "L1": "<+>",
        "COSINE": "<=>",
        "INNER_PRODUCT": "<#>",
    }

    # reference: https://github.com/pgvector/pgvector?tab=readme-ov-file#hnsw
    # OGX supports only index operator classes that are applied for embeddings with vector type
    PGVECTOR_DISTANCE_METRIC_TO_INDEX_OPERATOR_CLASS: dict[str, str] = {
        "L2": "vector_l2_ops",
        "L1": "vector_l1_ops",
        "COSINE": "vector_cosine_ops",
        "INNER_PRODUCT": "vector_ip_ops",
    }

    # pgvector's maximum embedding dimension for HNSW/IVFFlat indexes on column with type vector
    # references: https://github.com/pgvector/pgvector?tab=readme-ov-file#hnsw and https://github.com/pgvector/pgvector?tab=readme-ov-file#ivfflat
    MAX_EMBEDDING_DIMENSION_FOR_HNSW_AND_IVFFLAT_INDEX = 2000

    def __init__(
        self,
        vector_store: VectorStore,
        dimension: int,
        pool: asyncpg.Pool,
        distance_metric: str,
        vector_index: PGVectorIndexConfig,
        kvstore: KVStore | None = None,
    ):
        self.vector_store = vector_store
        self.dimension = dimension
        self.pool = pool
        self.kvstore = kvstore
        self.check_distance_metric_availability(distance_metric)
        self.distance_metric = distance_metric
        self.vector_index = vector_index
        self.table_name = None

    async def initialize(self) -> None:
        try:
            async with self.pool.acquire() as conn:
                # Sanitize the table name by replacing hyphens with underscores
                # SQL doesn't allow hyphens in table names, and vector_store.identifier may contain hyphens
                # when created with patterns like "test-vector-db-{uuid4()}"
                sanitized_identifier = sanitize_collection_name(self.vector_store.identifier)
                self.table_name = f"vs_{sanitized_identifier}"
                self._quoted_table = _quote_ident(self.table_name)

                await conn.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {self._quoted_table} (
                        id TEXT PRIMARY KEY,
                        document JSONB,
                        embedding vector({self.dimension}),
                        content_text TEXT,
                        tokenized_content TSVECTOR
                    )
                    """
                )

                # pgvector's embedding dimensions requirement to create an index for Approximate Nearest Neighbor (ANN) search is up to 2,000 dimensions for column with type vector
                if self.dimension <= self.MAX_EMBEDDING_DIMENSION_FOR_HNSW_AND_IVFFLAT_INDEX:
                    if self.vector_index.type == PGVectorIndexType.HNSW:
                        await self.create_hnsw_vector_index(conn)

                    # Create the index only after the table has some data (https://github.com/pgvector/pgvector?tab=readme-ov-file#ivfflat)
                    elif (
                        self.vector_index.type == PGVectorIndexType.IVFFlat
                        and not await self.check_conflicting_vector_index_exists(conn)
                    ):
                        log.info(
                            f"Creation of {PGVectorIndexType.IVFFlat} vector index in vector_store: {self.vector_store.identifier} was deferred. It will be created when the table has some data."
                        )

                else:
                    log.info(
                        f"Skip creation of {self.vector_index.type} vector index for embedding in PGVector for vector_store: {self.vector_store.identifier}"
                    )
                    log.info(
                        "PGVector requires embedding dimensions are up to 2,000 to successfully create a vector index."
                    )

                await self.create_gin_index(conn)

        except Exception as e:
            log.exception(f"Error creating PGVectorIndex for vector_store: {self.vector_store.identifier}")
            raise RuntimeError(f"Error creating PGVectorIndex for vector_store: {self.vector_store.identifier}") from e

    async def add_chunks(self, chunks: list[EmbeddedChunk]):
        if not chunks:
            return

        values = []
        for chunk in chunks:
            content_text = interleaved_content_as_str(chunk.content)
            values.append(
                (
                    f"{chunk.chunk_id}",
                    json.dumps(chunk.model_dump()),
                    chunk.embedding,
                    content_text,
                    content_text,
                )
            )

        async with self.pool.acquire() as conn:
            await conn.executemany(
                f"""
                INSERT INTO {self._quoted_table} (id, document, embedding, content_text, tokenized_content)
                VALUES ($1, $2::jsonb, $3::vector, $4, to_tsvector('english', $5))
                ON CONFLICT (id) DO UPDATE SET
                    embedding = EXCLUDED.embedding,
                    document = EXCLUDED.document,
                    content_text = EXCLUDED.content_text,
                    tokenized_content = EXCLUDED.tokenized_content
                """,
                values,
            )

            # Create the IVFFlat index only after the table has some data (https://github.com/pgvector/pgvector?tab=readme-ov-file#ivfflat)
            if (
                self.vector_index.type == PGVectorIndexType.IVFFlat
                and self.dimension <= self.MAX_EMBEDDING_DIMENSION_FOR_HNSW_AND_IVFFLAT_INDEX
            ):
                await self.create_ivfflat_vector_index(conn)

    async def query_vector(
        self, embedding: NDArray, k: int, score_threshold: float, filters: Any = None
    ) -> QueryChunksResponse:
        """Performs vector similarity search using PostgreSQL's search function. Default distance metric is COSINE.

        Args:
            embedding: The query embedding vector
            k: Number of results to return
            score_threshold: Minimum similarity score threshold
            filters: Optional filters for metadata-based filtering

        Returns:
            QueryChunksResponse with combined results
        """
        filter_clause, filter_params, next_idx = self._translate_filters(filters, param_idx=2)

        pgvector_search_function = self.get_pgvector_search_function()

        async with self.pool.acquire() as conn:
            # Specify the number of probes to allow PGVector to use Index Scan using IVFFlat index if it was configured (https://github.com/pgvector/pgvector?tab=readme-ov-file#query-options-1)
            if self.vector_index.type == PGVectorIndexType.IVFFlat:
                await conn.execute(f"SET ivfflat.probes = {self.vector_index.probes}")

            # Specify the max size of max heap that holds best candidates when traversing the graph (https://github.com/pgvector/pgvector?tab=readme-ov-file#query-options)
            elif self.vector_index.type == PGVectorIndexType.HNSW:
                await conn.execute(f"SET hnsw.ef_search = {self.vector_index.ef_search}")

            where = f"WHERE {filter_clause}" if filter_clause else ""

            rows = await conn.fetch(
                f"""
                SELECT document, embedding {pgvector_search_function} $1::vector AS distance
                FROM {self._quoted_table}
                {where}
                ORDER BY distance
                LIMIT ${next_idx}
                """,
                embedding.tolist(),
                *filter_params,
                k,
            )

            chunks = []
            scores = []
            for row in rows:
                dist = row["distance"]
                score = 1.0 / float(dist) if dist != 0 else float("inf")
                if score < score_threshold:
                    continue
                doc = row["document"]
                if isinstance(doc, str):
                    doc = json.loads(doc)
                chunks.append(load_embedded_chunk_with_backward_compat(doc))
                scores.append(score)

            return QueryChunksResponse(chunks=chunks, scores=scores)

    async def query_keyword(
        self,
        query_string: str,
        k: int,
        score_threshold: float,
        filters: ComparisonFilter | CompoundFilter | None = None,
    ) -> QueryChunksResponse:
        """Performs keyword-based search using PostgreSQL's full-text search with ts_rank scoring.

        Args:
            query_string: The text query for keyword search
            k: Number of results to return
            score_threshold: Minimum similarity score threshold
            filters: Optional filters for metadata-based filtering

        Returns:
            QueryChunksResponse with combined results
        """
        filter_clause, filter_params, next_idx = self._translate_filters(filters, param_idx=3)

        async with self.pool.acquire() as conn:
            filter_sql = f" AND ({filter_clause})" if filter_clause else ""

            rows = await conn.fetch(
                f"""
                SELECT document, ts_rank(tokenized_content, plainto_tsquery('english', $1)) AS score
                FROM {self._quoted_table}
                WHERE tokenized_content @@ plainto_tsquery('english', $2){filter_sql}
                ORDER BY score DESC
                LIMIT ${next_idx}
                """,
                query_string,
                query_string,
                *filter_params,
                k,
            )

            chunks = []
            scores = []
            for row in rows:
                score = row["score"]
                if score < score_threshold:
                    continue
                doc = row["document"]
                if isinstance(doc, str):
                    doc = json.loads(doc)
                chunks.append(load_embedded_chunk_with_backward_compat(doc))
                scores.append(float(score))

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
        """Hybrid search combining vector similarity and keyword search using configurable reranking.

        Args:
            embedding: The query embedding vector
            query_string: The text query for keyword search
            k: Number of results to return
            score_threshold: Minimum similarity score threshold
            reranker_type: Type of reranker to use ("rrf" or "weighted")
            reranker_params: Parameters for the reranker
            filters: Optional filters for metadata-based filtering

        Returns:
            QueryChunksResponse with combined results
        """
        if reranker_params is None:
            reranker_params = {}

        # Get results from both search methods, passing filters through
        vector_response = await self.query_vector(embedding, k, score_threshold, filters)
        keyword_response = await self.query_keyword(query_string, k, score_threshold, filters)

        # Convert responses to score dictionaries using chunk_id
        vector_scores = {
            chunk.chunk_id: score for chunk, score in zip(vector_response.chunks, vector_response.scores, strict=False)
        }
        keyword_scores = {
            chunk.chunk_id: score
            for chunk, score in zip(keyword_response.chunks, keyword_response.scores, strict=False)
        }

        # Combine scores using the reranking utility
        combined_scores = WeightedInMemoryAggregator.combine_search_results(
            vector_scores, keyword_scores, reranker_type, reranker_params
        )

        # Efficient top-k selection because it only tracks the k best candidates it's seen so far
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

    def _translate_filters(
        self, filters: ComparisonFilter | CompoundFilter | None, param_idx: int = 1
    ) -> tuple[str, list[Any], int]:
        """Translate OpenAI-compatible filters to PostgreSQL WHERE clause with $N parameters.

        Args:
            filters: The filter to translate (ComparisonFilter or CompoundFilter)
            param_idx: Starting parameter index for $N placeholders

        Returns:
            A tuple of (where_clause, parameters, next_idx) where where_clause is a SQL condition string,
            parameters is a list of values to be bound, and next_idx is the next available parameter index.
        """
        if filters is None:
            return "", [], param_idx

        return self._translate_single_filter(filters, param_idx)

    def _translate_single_filter(
        self, filter_obj: ComparisonFilter | CompoundFilter, param_idx: int
    ) -> tuple[str, list[Any], int]:
        """Translate a single filter to SQL."""
        if isinstance(filter_obj, ComparisonFilter):
            return self._translate_comparison_filter(filter_obj, param_idx)
        elif isinstance(filter_obj, CompoundFilter):
            return self._translate_compound_filter(filter_obj, param_idx)
        else:
            raise ValueError(f"Unknown filter type: {type(filter_obj)}")

    def _translate_comparison_filter(self, filter_obj: ComparisonFilter, param_idx: int) -> tuple[str, list[Any], int]:
        """Translate a comparison filter to PostgreSQL WHERE clause using JSONB operators."""
        key, value, op_type = filter_obj.key, filter_obj.value, filter_obj.type

        # Validate key to prevent SQL injection in JSONB path
        if not _VALID_KEY_PATTERN.match(key):
            raise ValueError(f"Invalid metadata key name: {key!r}")

        # Use ->> to extract metadata value as text from the JSONB document column
        expr = f"document->'metadata'->>'{key}'"

        if op_type in _PG_SQL_OPS:
            sql_op = _PG_SQL_OPS[op_type]
            # Check bool before int since bool is a subclass of int in Python
            if isinstance(value, bool):
                return f"({expr})::boolean {sql_op} ${param_idx}", [value], param_idx + 1
            elif isinstance(value, int | float):
                return f"({expr})::numeric {sql_op} ${param_idx}", [value], param_idx + 1
            else:
                return f"{expr} {sql_op} ${param_idx}", [value], param_idx + 1
        elif op_type == "in":
            if not isinstance(value, list):
                raise ValueError(f"'in' filter requires a list value, got {type(value)}")
            placeholders = ", ".join(f"${param_idx + i}" for i in range(len(value)))
            return f"{expr} IN ({placeholders})", [str(v) for v in value], param_idx + len(value)
        elif op_type == "nin":
            if not isinstance(value, list):
                raise ValueError(f"'nin' filter requires a list value, got {type(value)}")
            placeholders = ", ".join(f"${param_idx + i}" for i in range(len(value)))
            return f"{expr} NOT IN ({placeholders})", [str(v) for v in value], param_idx + len(value)
        else:
            raise ValueError(f"Unknown comparison operator: {op_type}")

    def _translate_compound_filter(self, filter_obj: CompoundFilter, param_idx: int) -> tuple[str, list[Any], int]:
        """Translate a compound filter (and/or) to PostgreSQL WHERE clause."""
        if not filter_obj.filters:
            return "", [], param_idx

        clauses = []
        params: list[Any] = []

        for sub_filter in filter_obj.filters:
            clause, sub_params, param_idx = self._translate_single_filter(sub_filter, param_idx)
            if clause:
                clauses.append(f"({clause})")
                params.extend(sub_params)

        if not clauses:
            return "", [], param_idx

        operator = " AND " if filter_obj.type == "and" else " OR "
        return operator.join(clauses), params, param_idx

    async def delete(self):
        async with self.pool.acquire() as conn:
            await conn.execute(f"DROP TABLE IF EXISTS {self._quoted_table}")

    async def delete_chunks(self, chunks_for_deletion: list[ChunkForDeletion]) -> None:
        """Remove chunks from the PostgreSQL table."""
        chunk_ids = [c.chunk_id for c in chunks_for_deletion]
        async with self.pool.acquire() as conn:
            await conn.execute(f"DELETE FROM {self._quoted_table} WHERE id = ANY($1::text[])", chunk_ids)

    def get_pgvector_index_operator_class(self) -> str:
        """Get the pgvector index operator class for the current distance metric.

        Returns:
            The operator class name.
        """
        return self.PGVECTOR_DISTANCE_METRIC_TO_INDEX_OPERATOR_CLASS[self.distance_metric]

    def get_pgvector_search_function(self) -> str:
        return self.PGVECTOR_DISTANCE_METRIC_TO_SEARCH_FUNCTION[self.distance_metric]

    def check_distance_metric_availability(self, distance_metric: str) -> None:
        """Check if the distance metric is supported by PGVector.

        Args:
            distance_metric: The distance metric to check

        Raises:
            ValueError: If the distance metric is not supported
        """
        if distance_metric not in self.PGVECTOR_DISTANCE_METRIC_TO_SEARCH_FUNCTION:
            supported_metrics = list(self.PGVECTOR_DISTANCE_METRIC_TO_SEARCH_FUNCTION.keys())
            raise ValueError(
                f"Distance metric '{distance_metric}' is not supported by PGVector. "
                f"Supported metrics are: {', '.join(supported_metrics)}"
            )

    async def create_hnsw_vector_index(self, conn: asyncpg.Connection) -> None:
        """Create PGVector HNSW vector index for Approximate Nearest Neighbor (ANN) search.

        Args:
            conn: asyncpg connection

        Raises:
            RuntimeError: If the error occurred when creating vector index in PGVector
        """
        # prevents from creating index for the table that already has conflicting index (HNSW or IVFFlat)
        if await self.check_conflicting_vector_index_exists(conn):
            return

        try:
            index_operator_class = self.get_pgvector_index_operator_class()
            index_name = _quote_ident(f"{self.table_name}_hnsw_idx")

            # Create HNSW (Hierarchical Navigable Small Worlds) index on embedding column to allow efficient and performant vector search in pgvector
            # HNSW finds the approximate nearest neighbors by only calculating distance metric for vectors it visits during graph traversal instead of processing all vectors
            await conn.execute(
                f"""
                CREATE INDEX IF NOT EXISTS {index_name}
                ON {self._quoted_table} USING hnsw(embedding {index_operator_class})
                WITH (m = {self.vector_index.m}, ef_construction = {self.vector_index.ef_construction});
                """
            )
            log.info(
                f"{PGVectorIndexType.HNSW} vector index was created with parameters m = {self.vector_index.m}, ef_construction = {self.vector_index.ef_construction} for vector_store: {self.vector_store.identifier}."
            )

        except asyncpg.PostgresError as e:
            raise RuntimeError(
                f"Failed to create {PGVectorIndexType.HNSW} vector index for vector_store: {self.vector_store.identifier}: {e}"
            ) from e

    async def create_ivfflat_vector_index(self, conn: asyncpg.Connection) -> None:
        """Create PGVector IVFFlat vector index for Approximate Nearest Neighbor (ANN) search.

        Args:
            conn: asyncpg connection

        Raises:
            RuntimeError: If the error occurred when creating vector index in PGVector
        """
        # prevents from creating index for the table that already has conflicting index (HNSW or IVFFlat)
        if await self.check_conflicting_vector_index_exists(conn):
            return

        # don't create index too early as it decreases a performance (https://github.com/pgvector/pgvector?tab=readme-ov-file#ivfflat)
        # create IVFFLAT index only if vector store has rows >= lists * 1000
        if await self.fetch_number_of_records(conn) < self.vector_index.lists * 1000:
            log.info(
                f"IVFFlat index wasn't created for vector_store {self.vector_store.identifier} because table doesn't have enough records."
            )
            return

        try:
            index_operator_class = self.get_pgvector_index_operator_class()
            index_name = _quote_ident(f"{self.table_name}_ivfflat_idx")

            # Create Inverted File with Flat Compression (IVFFlat) index on embedding column to allow efficient and performant vector search in pgvector
            # IVFFlat index divides vectors into lists, and then searches a subset of those lists that are closest to the query vector
            # Index should be created only after the table has some data (https://github.com/pgvector/pgvector?tab=readme-ov-file#ivfflat)
            await conn.execute(
                f"""
                CREATE INDEX IF NOT EXISTS {index_name}
                ON {self._quoted_table} USING ivfflat(embedding {index_operator_class})
                WITH (lists = {self.vector_index.lists});
                """
            )
            log.info(
                f"{PGVectorIndexType.IVFFlat} vector index was created with parameter lists = {self.vector_index.lists} for vector_store: {self.vector_store.identifier}."
            )

        except asyncpg.PostgresError as e:
            raise RuntimeError(
                f"Failed to create {PGVectorIndexType.IVFFlat} vector index for vector_store: {self.vector_store.identifier}: {e}"
            ) from e

    async def check_conflicting_vector_index_exists(self, conn: asyncpg.Connection) -> bool:
        """Check if vector index of any type has already been created for the table to prevent the conflict.

        Args:
            conn: asyncpg connection

        Returns:
            True if exists, otherwise False

        Raises:
            RuntimeError: If the error occurred when checking vector index exists in PGVector
        """
        try:
            log.info(
                f"Checking vector_store: {self.vector_store.identifier} for conflicting vector index in PGVector..."
            )
            result = await conn.fetchrow(
                """
                SELECT indexname FROM pg_indexes
                WHERE (indexname LIKE $1 OR indexname LIKE $2) AND tablename = $3;
                """,
                "%hnsw%",
                "%ivfflat%",
                self.table_name,
            )

            if result:
                log.warning(
                    f"Conflicting vector index {result['indexname']} already exists in vector_store: {self.vector_store.identifier}"
                )
                log.warning(
                    f"vector_store: {self.vector_store.identifier} will continue to use vector index {result['indexname']} to preserve performance."
                )
                return True

            log.info(f"vector_store: {self.vector_store.identifier} currently doesn't have conflicting vector index")
            log.info(f"Proceeding with creation of vector index for {self.vector_store.identifier}")
            return False

        except asyncpg.PostgresError as e:
            raise RuntimeError(f"Failed to check if vector index exists in PGVector: {e}") from e

    async def fetch_number_of_records(self, conn: asyncpg.Connection) -> int:
        """Returns number of records in a vector store.

        Args:
            conn: asyncpg connection

        Returns:
            number of records in a vector store

        Raises:
            RuntimeError: If the error occurred when fetching a number of records in a vector store in PGVector
        """
        try:
            log.info(f"Fetching number of records in vector_store: {self.vector_store.identifier}...")
            count = await conn.fetchval(f"SELECT COUNT(DISTINCT id) FROM {self._quoted_table}")

            if count:
                log.info(f"vector_store: {self.vector_store.identifier} has {count} records.")
                return count

            log.info(f"vector_store: {self.vector_store.identifier} currently doesn't have any records.")
            return 0

        except asyncpg.PostgresError as e:
            raise RuntimeError(f"Failed to check if vector store has records in PGVector: {e}") from e

    async def create_gin_index(self, conn: asyncpg.Connection) -> None:
        """Create GIN index for full-text search performance.

        Args:
            conn: asyncpg connection

        Raises:
            RuntimeError: If the error occurred when creating GIN index
        """
        try:
            index_name = _quote_ident(f"{self.table_name}_content_gin_idx")
            await conn.execute(
                f"""
                CREATE INDEX IF NOT EXISTS {index_name}
                ON {self._quoted_table} USING GIN(tokenized_content)
                """
            )
            log.info(f"GIN index verified for vector_store: {self.vector_store.identifier}.")

        except asyncpg.PostgresError as e:
            raise RuntimeError(
                f"Failed to create GIN index for vector_store: {self.vector_store.identifier}: {e}"
            ) from e


class PGVectorVectorIOAdapter(OpenAIVectorStoreMixin, VectorIO, VectorStoresProtocolPrivate):
    """Vector I/O adapter for PostgreSQL with pgvector."""

    def __init__(
        self,
        config: PGVectorVectorIOConfig,
        inference_api: Inference,
        files_api: Files | None = None,
        file_processor_api: FileProcessors | None = None,
        policy: list | None = None,
    ) -> None:
        super().__init__(
            inference_api=inference_api, files_api=files_api, kvstore=None, file_processor_api=file_processor_api
        )
        self.config = config
        self.pool = None
        self._pool_initialized = False
        self._pool_lock = asyncio.Lock()
        self.cache = {}
        self.vector_store_table = None
        self.metadata_collection_name = "openai_vector_stores_metadata"
        self._policy = policy or []

    @staticmethod
    async def _init_connection(conn: asyncpg.Connection) -> None:
        """Pool init callback — registers pgvector type codec on each new connection."""
        await register_vector(conn)

    @staticmethod
    async def _reset_connection(_conn: asyncpg.Connection) -> None:
        """No-op pool reset — asyncpg's default DEALLOCATE ALL destroys pgvector's custom binary type codecs.

        Skipping reset is safe: SET commands are re-applied per query, and
        type codecs persist correctly across connection reuses.
        """

    async def _ensure_pool(self) -> asyncpg.Pool:
        """Create pool lazily on first use to bind to the current event loop."""
        async with self._pool_lock:
            if self.pool is not None:
                try:
                    async with self.pool.acquire() as conn:
                        await conn.fetchval("SELECT 1")
                    return self.pool
                except Exception:
                    log.warning("Recreating connection pool — previous pool is not usable in current event loop")
                    try:
                        await self.pool.close()
                    except Exception:
                        log.debug("Failed to close stale connection pool during recreation")
                    self.pool = None
                    self._pool_initialized = False

            pool = await asyncpg.create_pool(
                host=self.config.host,
                port=self.config.port,
                database=self.config.db,
                user=self.config.user,
                password=self.config.password,
                min_size=self.config.pool_min_size,
                max_size=self.config.pool_max_size,
                statement_cache_size=self.config.statement_cache_size,
                init=self._init_connection,
                reset=self._reset_connection,
            )

            try:
                if not self._pool_initialized:
                    async with pool.acquire() as conn:
                        version = await check_extension_version(conn)
                        if version:
                            log.info("Vector extension version", version=version)
                        else:
                            await create_vector_extension(conn)

                        await conn.execute(
                            """
                            CREATE TABLE IF NOT EXISTS metadata_store (
                                key TEXT PRIMARY KEY,
                                data JSONB
                            )
                            """
                        )
                    self._pool_initialized = True
            except Exception:
                await pool.close()
                raise

            self.pool = pool
            return self.pool

    async def initialize(self) -> None:
        safe_config = {**self.config.model_dump(exclude={"password"}), "password": "******"}
        log.info(f"Initializing PGVector memory adapter with config: {safe_config}")
        self.kvstore = await kvstore_impl(self.config.persistence)

        if self.config.metadata_store:
            from ogx.core.storage.sqlstore import authorized_sqlstore

            self.metadata_store = await authorized_sqlstore(self.config.metadata_store, self._policy)

        await self.initialize_openai_vector_stores()

        try:
            pool = await self._ensure_pool()
        except Exception as e:
            log.exception("Could not connect to PGVector database server")
            raise RuntimeError("Could not connect to PGVector database server") from e

        try:
            # Load existing vector stores from KV store into cache
            start_key = VECTOR_DBS_PREFIX
            end_key = f"{VECTOR_DBS_PREFIX}\xff"
            stored_vector_stores = await self.kvstore.values_in_range(start_key, end_key)
            for vector_store_data in stored_vector_stores:
                vector_store = VectorStore.model_validate_json(vector_store_data)
                pgvector_index = PGVectorIndex(
                    vector_store=vector_store,
                    dimension=vector_store.embedding_dimension,
                    pool=pool,
                    kvstore=self.kvstore,
                    distance_metric=self.config.distance_metric,
                    vector_index=self.config.vector_index,
                )
                await pgvector_index.initialize()
                index = VectorStoreWithIndex(vector_store, index=pgvector_index, inference_api=self.inference_api)
                self.cache[vector_store.identifier] = index
        except Exception:
            await self.shutdown()
            raise

    async def shutdown(self) -> None:
        if self.pool is not None:
            try:
                await self.pool.close()
                log.info("Connection pool to PGVector database server closed")
            except Exception:
                log.exception("Failed to close PGVector connection pool")
        # Clean up mixin resources (file batch tasks)
        await super().shutdown()

    async def register_vector_store(self, vector_store: VectorStore) -> None:
        # Persist vector DB metadata in the KV store
        if self.kvstore is None:
            raise RuntimeError("KVStore not initialized. Call initialize() before registering vector stores.")

        pool = await self._ensure_pool()

        # Save to kvstore for persistence
        key = f"{VECTOR_DBS_PREFIX}{vector_store.identifier}"
        await self.kvstore.set(key=key, value=vector_store.model_dump_json())

        # Upsert model metadata in Postgres
        await upsert_models(pool, [(vector_store.identifier, vector_store)])

        # Create and cache the PGVector index table for the vector DB
        pgvector_index = PGVectorIndex(
            vector_store=vector_store,
            dimension=vector_store.embedding_dimension,
            pool=pool,
            kvstore=self.kvstore,
            distance_metric=self.config.distance_metric,
            vector_index=self.config.vector_index,
        )
        await pgvector_index.initialize()
        index = VectorStoreWithIndex(vector_store, index=pgvector_index, inference_api=self.inference_api)
        self.cache[vector_store.identifier] = index

    async def unregister_vector_store(self, vector_store_id: str) -> None:
        # Remove provider index and cache
        if vector_store_id in self.cache:
            await self.cache[vector_store_id].index.delete()
            del self.cache[vector_store_id]

        # Delete vector DB metadata from KV store
        if self.kvstore is None:
            raise RuntimeError("KVStore not initialized. Call initialize() before unregistering vector stores.")
        await self.kvstore.delete(key=f"{VECTOR_DBS_PREFIX}{vector_store_id}")

        # Delete vector store metadata from PGVector metadata_store table
        pool = await self._ensure_pool()
        await remove_vector_store_metadata(pool, vector_store_id)

    async def insert_chunks(self, request: InsertChunksRequest) -> None:
        index = await self._get_and_cache_vector_store_index(request.vector_store_id)
        await index.insert_chunks(request)

    async def query_chunks(self, request: QueryChunksRequest) -> QueryChunksResponse:
        index = await self._get_and_cache_vector_store_index(request.vector_store_id)
        return await index.query_chunks(request)

    async def _get_and_cache_vector_store_index(self, vector_store_id: str) -> VectorStoreWithIndex:
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
        pool = await self._ensure_pool()
        index = PGVectorIndex(
            vector_store,
            vector_store.embedding_dimension,
            pool,
            distance_metric=self.config.distance_metric,
            vector_index=self.config.vector_index,
        )
        await index.initialize()
        self.cache[vector_store_id] = VectorStoreWithIndex(vector_store, index, self.inference_api)
        return self.cache[vector_store_id]

    async def delete_chunks(self, request: DeleteChunksRequest) -> None:
        """Delete a chunk from a PostgreSQL vector store."""
        index = await self._get_and_cache_vector_store_index(request.vector_store_id)
        if not index:
            raise VectorStoreNotFoundError(request.vector_store_id)

        await index.index.delete_chunks(request.chunks)
