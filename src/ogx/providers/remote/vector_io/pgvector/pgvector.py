# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import heapq
import re
from typing import Any

import psycopg2
from numpy.typing import NDArray
from psycopg2 import sql
from psycopg2.extensions import cursor
from psycopg2.extras import Json, execute_values
from pydantic import BaseModel, TypeAdapter

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


def check_extension_version(cur):
    """Query the installed pgvector extension version.

    Args:
        cur: database cursor

    Returns:
        Version string if the extension is installed, otherwise None
    """
    cur.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
    result = cur.fetchone()
    return result[0] if result else None


def create_vector_extension(cur: cursor) -> None:
    """Create the pgvector extension in the database.

    Args:
        cur: database cursor
    """
    try:
        log.info("Vector extension not found, creating...")
        cur.execute("CREATE EXTENSION vector;")
        log.info("Vector extension created successfully")
        log.info(f"Vector extension version: {check_extension_version(cur)}")

    except psycopg2.Error as e:
        raise RuntimeError(f"Failed to create vector extension for PGVector: {e}") from e


def upsert_models(conn, keys_models: list[tuple[str, BaseModel]]):
    """Insert or update serialized Pydantic models in the metadata_store table.

    Args:
        conn: active PostgreSQL connection
        keys_models: list of (key, model) tuples to upsert
    """
    with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
        query = sql.SQL(
            """
            INSERT INTO metadata_store (key, data)
            VALUES %s
            ON CONFLICT (key) DO UPDATE
            SET data = EXCLUDED.data
        """
        )

        values = [(key, Json(model.model_dump())) for key, model in keys_models]
        execute_values(cur, query, values, template="(%s, %s)")


def remove_vector_store_metadata(conn: psycopg2.extensions.connection, vector_store_id: str) -> None:
    """
    Performs removal of vector store metadata from PGVector metadata_store table when vector store is unregistered

    Args:
        conn: active PostgreSQL connection
        vector_store_id: identifier of VectorStore resource
    """
    try:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM metadata_store WHERE key = %s", (vector_store_id,))
            if cur.rowcount > 0:
                log.info(f"Removed metadata for vector store '{vector_store_id}' from PGVector metadata_store table.")

    except Exception as e:
        raise RuntimeError(
            f"Error removing metadata from PGVector metadata_store for vector_store: {vector_store_id}"
        ) from e


def load_models(cur, cls):
    """Load and deserialize all models from the metadata_store table.

    Args:
        cur: database cursor
        cls: Pydantic model class to deserialize into

    Returns:
        List of validated model instances
    """
    cur.execute("SELECT key, data FROM metadata_store")
    rows = cur.fetchall()
    return [TypeAdapter(cls).validate_python(row["data"]) for row in rows]


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
        conn: psycopg2.extensions.connection,
        distance_metric: str,
        vector_index: PGVectorIndexConfig,
        kvstore: KVStore | None = None,
    ):
        self.vector_store = vector_store
        self.dimension = dimension
        self.conn = conn
        self.kvstore = kvstore
        self.check_distance_metric_availability(distance_metric)
        self.distance_metric = distance_metric
        self.vector_index = vector_index
        self.table_name = None

    async def initialize(self) -> None:
        try:
            with self.conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
                # Sanitize the table name by replacing hyphens with underscores
                # SQL doesn't allow hyphens in table names, and vector_store.identifier may contain hyphens
                # when created with patterns like "test-vector-db-{uuid4()}"
                sanitized_identifier = sanitize_collection_name(self.vector_store.identifier)
                self.table_name = f"vs_{sanitized_identifier}"
                self._table_sql = sql.Identifier(self.table_name)

                cur.execute(
                    sql.SQL(
                        """
                    CREATE TABLE IF NOT EXISTS {} (
                        id TEXT PRIMARY KEY,
                        document JSONB,
                        embedding vector({}),
                        content_text TEXT,
                        tokenized_content TSVECTOR
                    )
                """
                    ).format(self._table_sql, sql.Literal(self.dimension))
                )

                # pgvector's embedding dimensions requirement to create an index for Approximate Nearest Neighbor (ANN) search is up to 2,000 dimensions for column with type vector
                if self.dimension <= self.MAX_EMBEDDING_DIMENSION_FOR_HNSW_AND_IVFFLAT_INDEX:
                    if self.vector_index.type == PGVectorIndexType.HNSW:
                        await self.create_hnsw_vector_index(cur)

                    # Create the index only after the table has some data (https://github.com/pgvector/pgvector?tab=readme-ov-file#ivfflat)
                    elif (
                        self.vector_index.type == PGVectorIndexType.IVFFlat
                        and not await self.check_conflicting_vector_index_exists(cur)
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

                await self.create_gin_index(cur)

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
                    Json(chunk.model_dump()),
                    chunk.embedding,  # Already a list[float]
                    content_text,
                    content_text,  # Pass content_text twice - once for content_text column, once for to_tsvector function. Eg. to_tsvector(content_text) = tokenized_content
                )
            )

        query = sql.SQL(
            """
        INSERT INTO {} (id, document, embedding, content_text, tokenized_content)
        VALUES %s
        ON CONFLICT (id) DO UPDATE SET
            embedding = EXCLUDED.embedding,
            document = EXCLUDED.document,
            content_text = EXCLUDED.content_text,
            tokenized_content = EXCLUDED.tokenized_content
    """
        ).format(self._table_sql)
        with self.conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            execute_values(cur, query, values, template="(%s, %s, %s::vector, %s, to_tsvector('english', %s))")

            # Create the IVFFlat index only after the table has some data (https://github.com/pgvector/pgvector?tab=readme-ov-file#ivfflat)
            if (
                self.vector_index.type == PGVectorIndexType.IVFFlat
                and self.dimension <= self.MAX_EMBEDDING_DIMENSION_FOR_HNSW_AND_IVFFLAT_INDEX
            ):
                await self.create_ivfflat_vector_index(cur)

    async def query_vector(
        self, embedding: NDArray, k: int, score_threshold: float, filters: Any = None
    ) -> QueryChunksResponse:
        """
        Performs vector similarity search using PostgreSQL's search function. Default distance metric is COSINE.

        Args:
            embedding: The query embedding vector
            k: Number of results to return
            score_threshold: Minimum similarity score threshold
            filters: Optional filters for metadata-based filtering

        Returns:
            QueryChunksResponse with combined results
        """
        filter_clause, filter_params = self._translate_filters(filters)

        pgvector_search_function = self.get_pgvector_search_function()

        with self.conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            # Specify the number of probes to allow PGVector to use Index Scan using IVFFlat index if it was configured (https://github.com/pgvector/pgvector?tab=readme-ov-file#query-options-1)
            if self.vector_index.type == PGVectorIndexType.IVFFlat:
                cur.execute(
                    f"""
                    SET ivfflat.probes = {self.vector_index.probes};
                """
                )

            # Specify the max size of max heap that holds best candidates when traversing the graph (https://github.com/pgvector/pgvector?tab=readme-ov-file#query-options)
            elif self.vector_index.type == PGVectorIndexType.HNSW:
                cur.execute(
                    f"""
                    SET hnsw.ef_search = {self.vector_index.ef_search};
                """
                )

            where = sql.SQL("WHERE {}").format(sql.SQL(filter_clause)) if filter_clause else sql.SQL("")

            cur.execute(
                sql.SQL(
                    """
            SELECT document, embedding {} %s::vector AS distance
            FROM {}
            {}
            ORDER BY distance
            LIMIT %s
        """
                ).format(sql.SQL(pgvector_search_function), self._table_sql, where),
                (embedding.tolist(), *filter_params, k),
            )
            results = cur.fetchall()

            chunks = []
            scores = []
            for doc, dist in results:
                score = 1.0 / float(dist) if dist != 0 else float("inf")
                if score < score_threshold:
                    continue
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
        """
        Performs keyword-based search using PostgreSQL's full-text search with ts_rank scoring.

        Args:
            query_string: The text query for keyword search
            k: Number of results to return
            score_threshold: Minimum similarity score threshold

        Returns:
            QueryChunksResponse with combined results
        """
        filter_clause, filter_params = self._translate_filters(filters)

        with self.conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            # Use plainto_tsquery to handle user input safely and ts_rank for relevance scoring
            filter_sql = sql.SQL(" AND ({})").format(sql.SQL(filter_clause)) if filter_clause else sql.SQL("")

            cur.execute(
                sql.SQL(
                    """
            SELECT document, ts_rank(tokenized_content, plainto_tsquery('english', %s)) AS score
            FROM {}
            WHERE tokenized_content @@ plainto_tsquery('english', %s){}
            ORDER BY score DESC
            LIMIT %s
        """
                ).format(self._table_sql, filter_sql),
                (query_string, query_string, *filter_params, k),
            )
            results = cur.fetchall()

            chunks = []
            scores = []
            for doc, score in results:
                if score < score_threshold:
                    continue
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
        """
        Hybrid search combining vector similarity and keyword search using configurable reranking.

        Args:
            embedding: The query embedding vector
            query_string: The text query for keyword search
            k: Number of results to return
            score_threshold: Minimum similarity score threshold
            reranker_type: Type of reranker to use ("rrf" or "weighted")
            reranker_params: Parameters for the reranker

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

    def _translate_filters(self, filters: ComparisonFilter | CompoundFilter | None) -> tuple[str, list[Any]]:
        """Translate OpenAI-compatible filters to PostgreSQL WHERE clause with parameters.

        Args:
            filters: The filter to translate (ComparisonFilter or CompoundFilter)

        Returns:
            A tuple of (where_clause, parameters) where where_clause is a SQL condition string
            and parameters is a list of values to be bound to the query.
        """
        if filters is None:
            return "", []

        return self._translate_single_filter(filters)

    def _translate_single_filter(self, filter_obj: ComparisonFilter | CompoundFilter) -> tuple[str, list[Any]]:
        """Translate a single filter to SQL."""
        if isinstance(filter_obj, ComparisonFilter):
            return self._translate_comparison_filter(filter_obj)
        elif isinstance(filter_obj, CompoundFilter):
            return self._translate_compound_filter(filter_obj)
        else:
            raise ValueError(f"Unknown filter type: {type(filter_obj)}")

    def _translate_comparison_filter(self, filter_obj: ComparisonFilter) -> tuple[str, list[Any]]:
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
                return f"({expr})::boolean {sql_op} %s", [value]
            elif isinstance(value, int | float):
                return f"({expr})::numeric {sql_op} %s", [value]
            else:
                return f"{expr} {sql_op} %s", [value]
        elif op_type == "in":
            if not isinstance(value, list):
                raise ValueError(f"'in' filter requires a list value, got {type(value)}")
            placeholders = ", ".join("%s" for _ in value)
            return f"{expr} IN ({placeholders})", [str(v) for v in value]
        elif op_type == "nin":
            if not isinstance(value, list):
                raise ValueError(f"'nin' filter requires a list value, got {type(value)}")
            placeholders = ", ".join("%s" for _ in value)
            return f"{expr} NOT IN ({placeholders})", [str(v) for v in value]
        else:
            raise ValueError(f"Unknown comparison operator: {op_type}")

    def _translate_compound_filter(self, filter_obj: CompoundFilter) -> tuple[str, list[Any]]:
        """Translate a compound filter (and/or) to PostgreSQL WHERE clause."""
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

    async def delete(self):
        with self.conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            cur.execute(sql.SQL("DROP TABLE IF EXISTS {}").format(self._table_sql))

    async def delete_chunks(self, chunks_for_deletion: list[ChunkForDeletion]) -> None:
        """Remove a chunk from the PostgreSQL table."""
        chunk_ids = [c.chunk_id for c in chunks_for_deletion]
        with self.conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            # Fix: Use proper tuple parameter binding with explicit array cast
            cur.execute(sql.SQL("DELETE FROM {} WHERE id = ANY(%s::text[])").format(self._table_sql), (chunk_ids,))

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

    async def create_hnsw_vector_index(self, cur: cursor) -> None:
        """Create PGVector HNSW vector index for Approximate Nearest Neighbor (ANN) search

        Args:
            cur: PostgreSQL cursor

        Raises:
            RuntimeError: If the error occurred when creating vector index in PGVector
        """

        # prevents from creating index for the table that already has conflicting index (HNSW or IVFFlat)
        if await self.check_conflicting_vector_index_exists(cur):
            return

        try:
            index_operator_class = self.get_pgvector_index_operator_class()

            # Create HNSW (Hierarchical Navigable Small Worlds) index on embedding column to allow efficient and performant vector search in pgvector
            # HNSW finds the approximate nearest neighbors by only calculating distance metric for vectors it visits during graph traversal instead of processing all vectors
            cur.execute(
                sql.SQL(
                    """
                CREATE INDEX IF NOT EXISTS {}
                ON {} USING hnsw(embedding {}) WITH (m = {}, ef_construction = {});
            """
                ).format(
                    sql.Identifier(f"{self.table_name}_hnsw_idx"),
                    self._table_sql,
                    sql.SQL(index_operator_class),
                    sql.Literal(self.vector_index.m),
                    sql.Literal(self.vector_index.ef_construction),
                )
            )
            log.info(
                f"{PGVectorIndexType.HNSW} vector index was created with parameters m = {self.vector_index.m}, ef_construction = {self.vector_index.ef_construction} for vector_store: {self.vector_store.identifier}."
            )

        except psycopg2.Error as e:
            raise RuntimeError(
                f"Failed to create {PGVectorIndexType.HNSW} vector index for vector_store: {self.vector_store.identifier}: {e}"
            ) from e

    async def create_ivfflat_vector_index(self, cur: cursor) -> None:
        """Create PGVector IVFFlat vector index for Approximate Nearest Neighbor (ANN) search

        Args:
            cur: PostgreSQL cursor

        Raises:
            RuntimeError: If the error occurred when creating vector index in PGVector
        """

        # prevents from creating index for the table that already has conflicting index (HNSW or IVFFlat)
        if await self.check_conflicting_vector_index_exists(cur):
            return

        # don't create index too early as it decreases a performance (https://github.com/pgvector/pgvector?tab=readme-ov-file#ivfflat)
        # create IVFFLAT index only if vector store has rows >= lists * 1000
        if await self.fetch_number_of_records(cur) < self.vector_index.lists * 1000:
            log.info(
                f"IVFFlat index wasn't created for vector_store {self.vector_store.identifier} because table doesn't have enough records."
            )
            return

        try:
            index_operator_class = self.get_pgvector_index_operator_class()

            # Create Inverted File with Flat Compression (IVFFlat) index on embedding column to allow efficient and performant vector search in pgvector
            # IVFFlat index divides vectors into lists, and then searches a subset of those lists that are closest to the query vector
            # Index should be created only after the table has some data (https://github.com/pgvector/pgvector?tab=readme-ov-file#ivfflat)
            cur.execute(
                sql.SQL(
                    """
                CREATE INDEX IF NOT EXISTS {}
                ON {} USING ivfflat(embedding {}) WITH (lists = {});
            """
                ).format(
                    sql.Identifier(f"{self.table_name}_ivfflat_idx"),
                    self._table_sql,
                    sql.SQL(index_operator_class),
                    sql.Literal(self.vector_index.lists),
                )
            )
            log.info(
                f"{PGVectorIndexType.IVFFlat} vector index was created with parameter lists = {self.vector_index.lists} for vector_store: {self.vector_store.identifier}."
            )

        except psycopg2.Error as e:
            raise RuntimeError(
                f"Failed to create {PGVectorIndexType.IVFFlat} vector index for vector_store: {self.vector_store.identifier}: {e}"
            ) from e

    async def check_conflicting_vector_index_exists(self, cur: cursor) -> bool:
        """Check if vector index of any type has already been created for the table to prevent the conflict

        Args:
            cur: PostgreSQL cursor

        Returns:
            True if exists, otherwise False

        Raises:
            RuntimeError: If the error occurred when checking vector index exists in PGVector
        """
        try:
            log.info(
                f"Checking vector_store: {self.vector_store.identifier} for conflicting vector index in PGVector..."
            )
            cur.execute(
                """
                SELECT indexname FROM pg_indexes
                WHERE (indexname LIKE %s OR indexname LIKE %s) AND tablename = %s;
                """,
                (
                    "%hnsw%",
                    "%ivfflat%",
                    self.table_name,
                ),
            )
            result = cur.fetchone()

            if result:
                log.warning(
                    f"Conflicting vector index {result[0]} already exists in vector_store: {self.vector_store.identifier}"
                )
                log.warning(
                    f"vector_store: {self.vector_store.identifier} will continue to use vector index {result[0]} to preserve performance."
                )
                return True

            log.info(f"vector_store: {self.vector_store.identifier} currently doesn't have conflicting vector index")
            log.info(f"Proceeding with creation of vector index for {self.vector_store.identifier}")
            return False

        except psycopg2.Error as e:
            raise RuntimeError(f"Failed to check if vector index exists in PGVector: {e}") from e

    async def fetch_number_of_records(self, cur: cursor) -> int:
        """Returns number of records in a vector store

        Args:
            cur: PostgreSQL cursor

        Returns:
            number of records in a vector store

        Raises:
            RuntimeError: If the error occurred when fetching a number of records in a vector store in PGVector
        """
        try:
            log.info(f"Fetching number of records in vector_store: {self.vector_store.identifier}...")
            cur.execute(
                sql.SQL(
                    """
                SELECT COUNT(DISTINCT id)
                FROM {};
                """
                ).format(self._table_sql)
            )
            result = cur.fetchone()

            if result:
                log.info(f"vector_store: {self.vector_store.identifier} has {result[0]} records.")
                return result[0]

            log.info(f"vector_store: {self.vector_store.identifier} currently doesn't have any records.")
            return 0

        except psycopg2.Error as e:
            raise RuntimeError(f"Failed to check if vector store has records in PGVector: {e}") from e

    async def create_gin_index(self, cur: cursor) -> None:
        """Create GIN index for full-text search performance

        Args:
            cur: PostgreSQL cursor

        Raises:
            RuntimeError: If the error occurred when creating GIN index
        """

        try:
            cur.execute(
                sql.SQL(
                    """
                CREATE INDEX IF NOT EXISTS {}
                ON {} USING GIN(tokenized_content)
            """
                ).format(
                    sql.Identifier(f"{self.table_name}_content_gin_idx"),
                    self._table_sql,
                )
            )
            log.info(f"GIN index verified for vector_store: {self.vector_store.identifier}.")

        except psycopg2.Error as e:
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
    ) -> None:
        super().__init__(
            inference_api=inference_api, files_api=files_api, kvstore=None, file_processor_api=file_processor_api
        )
        self.config = config
        self.conn = None
        self.cache = {}
        self.vector_store_table = None
        self.metadata_collection_name = "openai_vector_stores_metadata"

    async def initialize(self) -> None:
        # Create a safe config representation with masked password for logging
        safe_config = {**self.config.model_dump(exclude={"password"}), "password": "******"}
        log.info(f"Initializing PGVector memory adapter with config: {safe_config}")
        self.kvstore = await kvstore_impl(self.config.persistence)
        await self.initialize_openai_vector_stores()

        try:
            self.conn = psycopg2.connect(
                host=self.config.host,
                port=self.config.port,
                database=self.config.db,
                user=self.config.user,
                password=self.config.password,
            )
            self.conn.autocommit = True
            with self.conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
                version = check_extension_version(cur)
                if version:
                    log.info(f"Vector extension version: {version}")
                else:
                    create_vector_extension(cur)

                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS metadata_store (
                        key TEXT PRIMARY KEY,
                        data JSONB
                    )
                """
                )
        except Exception as e:
            log.exception("Could not connect to PGVector database server")
            raise RuntimeError("Could not connect to PGVector database server") from e

        # Load existing vector stores from KV store into cache
        start_key = VECTOR_DBS_PREFIX
        end_key = f"{VECTOR_DBS_PREFIX}\xff"
        stored_vector_stores = await self.kvstore.values_in_range(start_key, end_key)
        for vector_store_data in stored_vector_stores:
            vector_store = VectorStore.model_validate_json(vector_store_data)
            pgvector_index = PGVectorIndex(
                vector_store=vector_store,
                dimension=vector_store.embedding_dimension,
                conn=self.conn,
                kvstore=self.kvstore,
                distance_metric=self.config.distance_metric,
                vector_index=self.config.vector_index,
            )
            await pgvector_index.initialize()
            index = VectorStoreWithIndex(vector_store, index=pgvector_index, inference_api=self.inference_api)
            self.cache[vector_store.identifier] = index

    async def shutdown(self) -> None:
        if self.conn is not None:
            self.conn.close()
            log.info("Connection to PGVector database server closed")
        # Clean up mixin resources (file batch tasks)
        await super().shutdown()

    async def register_vector_store(self, vector_store: VectorStore) -> None:
        # Persist vector DB metadata in the KV store
        if self.kvstore is None:
            raise RuntimeError("KVStore not initialized. Call initialize() before registering vector stores.")

        # Save to kvstore for persistence
        key = f"{VECTOR_DBS_PREFIX}{vector_store.identifier}"
        await self.kvstore.set(key=key, value=vector_store.model_dump_json())

        # Upsert model metadata in Postgres
        upsert_models(self.conn, [(vector_store.identifier, vector_store)])

        # Create and cache the PGVector index table for the vector DB
        pgvector_index = PGVectorIndex(
            vector_store=vector_store,
            dimension=vector_store.embedding_dimension,
            conn=self.conn,
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
        remove_vector_store_metadata(self.conn, vector_store_id)

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
        index = PGVectorIndex(
            vector_store,
            vector_store.embedding_dimension,
            self.conn,
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
