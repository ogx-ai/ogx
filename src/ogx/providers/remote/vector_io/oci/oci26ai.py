# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import heapq
import json
from array import array
from typing import Any

import numpy as np
import oracledb
from numpy.typing import NDArray

from ogx.core.storage.kvstore import kvstore_impl
from ogx.log import get_logger
from ogx.providers.remote.vector_io.oci.config import OCI26aiVectorIOConfig
from ogx.providers.utils.memory.openai_vector_store_mixin import VERSION as OPENAIMIXINVERSION
from ogx.providers.utils.memory.openai_vector_store_mixin import OpenAIVectorStoreMixin
from ogx.providers.utils.memory.vector_store import (
    ChunkForDeletion,
    EmbeddingIndex,
    VectorStoreWithIndex,
)
from ogx.providers.utils.vector_io.filters import Filter
from ogx.providers.utils.vector_io.vector_utils import (
    WeightedInMemoryAggregator,
    sanitize_collection_name,
)
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

logger = get_logger(name=__name__, category="vector_io::oci26ai")

VERSION = "v1"
VECTOR_DBS_PREFIX = f"vector_stores:oci26ai:{VERSION}::"
VECTOR_INDEX_PREFIX = f"vector_index:oci26ai:{VERSION}::"
OPENAI_VECTOR_STORES_PREFIX = f"openai_vector_stores:oci26ai:{OPENAIMIXINVERSION}::"
OPENAI_VECTOR_STORES_FILES_PREFIX = f"openai_vector_stores_files:oci26ai:{OPENAIMIXINVERSION}::"
OPENAI_VECTOR_STORES_FILES_CONTENTS_PREFIX = f"openai_vector_stores_files_contents:oci26ai:{VERSION}::"


def normalize_embedding(embedding: np.typing.NDArray) -> np.typing.NDArray:
    """
    Normalize an embedding vector to unit length (L2 norm).

    This is required for cosine similarity to behave correctly.
    """
    if embedding is None:
        raise ValueError("Embedding cannot be None")

    emb = np.asarray(embedding, dtype=np.float64)

    norm = np.linalg.norm(emb)
    if norm == 0.0:
        raise ValueError("Cannot normalize zero-length vector")

    return emb / norm


class OCI26aiIndex(EmbeddingIndex):
    """Embedding index backed by Oracle 26ai vector database."""

    def __init__(
        self,
        connection,
        vector_store: VectorStore,
        consistency_level="Strong",
        kvstore: KVStore | None = None,
        vector_datatype: str = "FLOAT32",
    ):
        self.connection = connection
        self.vector_store = vector_store
        self.table_name = sanitize_collection_name(vector_store.vector_store_id)
        self.dimensions = vector_store.embedding_dimension
        self.consistency_level = consistency_level
        self.kvstore = kvstore
        self.vector_datatype = vector_datatype

    async def initialize(self) -> None:
        logger.info("Attempting to create table", table_name=self.table_name)
        cursor = self.connection.cursor()
        try:
            #  Create table
            create_table_sql = f"""
                CREATE TABLE IF NOT EXISTS {self.table_name} (
                    chunk_id VARCHAR2(100) PRIMARY KEY,
                    content CLOB,
                    vector VECTOR({self.dimensions}, {self.vector_datatype}),
                    metadata JSON,
                    chunk_metadata JSON
                );
            """
            logger.debug("Executing SQL", create_table_sql=create_table_sql)
            cursor.execute(create_table_sql)
            logger.info("Table created successfully", table_name=self.table_name)

            await self.create_indexes()
        finally:
            cursor.close()

    async def index_exists(self, index_name: str) -> bool:
        cursor = self.connection.cursor()
        try:
            cursor.execute(
                """
                SELECT COUNT(*)
                FROM USER_INDEXES
                WHERE INDEX_NAME = :index_name
            """,
                index_name=index_name.upper(),
            )
            (count,) = cursor.fetchone()
            return bool(count > 0)
        finally:
            cursor.close()

    async def create_indexes(self):
        indexes = [
            {
                "name": f"{self.table_name}_content_idx",
                "sql": f"""
                    CREATE INDEX {self.table_name}_CONTENT_IDX
                    ON {self.table_name}(content)
                    INDEXTYPE IS CTXSYS.CONTEXT
                    PARAMETERS ('SYNC (EVERY "FREQ=SECONDLY;INTERVAL=5")');
                """,
            },
            {
                "name": f"{self.table_name}_vector_ivf_idx",
                "sql": f"""
                    CREATE VECTOR INDEX {self.table_name}_vector_ivf_idx
                    ON {self.table_name}(vector)
                    ORGANIZATION NEIGHBOR PARTITIONS
                    DISTANCE COSINE
                    WITH TARGET ACCURACY 95
                """,
            },
        ]

        for idx in indexes:
            if not await self.index_exists(idx["name"]):
                logger.info("Creating index", name=idx["name"])
                cursor = self.connection.cursor()
                try:
                    cursor.execute(idx["sql"])
                    logger.info("Index created successfully", name=idx["name"])
                finally:
                    cursor.close()
            else:
                logger.info("Index already exists, skipping", name=idx["name"])

    async def add_chunks(self, embedded_chunks: list[EmbeddedChunk]):
        array_type = "d" if self.vector_datatype == "FLOAT64" else "f"
        data = []
        for chunk in embedded_chunks:
            chunk_step = chunk.model_dump()
            data.append(
                {
                    "chunk_id": chunk.chunk_id,
                    "content": chunk.content,
                    "vector": array(array_type, normalize_embedding(np.array(chunk.embedding)).astype(float).tolist()),
                    "metadata": json.dumps(chunk_step.get("metadata")),
                    "chunk_metadata": json.dumps(chunk_step.get("chunk_metadata")),
                }
            )
        cursor = self.connection.cursor()
        try:
            query = f"""
                MERGE INTO {self.table_name} t
                USING (
                    SELECT
                        :chunk_id       AS chunk_id,
                        :content        AS content,
                        :vector         AS vector,
                        :metadata       AS metadata,
                        :chunk_metadata AS chunk_metadata
                    FROM dual
                ) s
                ON (t.chunk_id = s.chunk_id)

                WHEN MATCHED THEN
                UPDATE SET
                    t.content           = s.content,
                    t.vector            = TO_VECTOR(s.vector, {self.dimensions}, {self.vector_datatype}),
                    t.metadata          = s.metadata,
                    t.chunk_metadata    = s.chunk_metadata

                WHEN NOT MATCHED THEN
                INSERT (chunk_id, content, vector, metadata, chunk_metadata)
                VALUES (s.chunk_id, s.content, TO_VECTOR(s.vector, {self.dimensions}, {self.vector_datatype}), s.metadata, s.chunk_metadata)
                """
            logger.debug("query", query=query)
            cursor.executemany(
                query,
                data,
            )
            logger.info("Merge completed successfully")
        except Exception as e:
            logger.error("Error inserting chunks into Oracle 26AI table", table_name=self.table_name, error=str(e))
            raise
        finally:
            cursor.close()

    async def query_vector(
        self,
        embedding: NDArray,
        k: int,
        score_threshold: float,
        filters: Filter | None = None,
    ) -> QueryChunksResponse:
        """
        Oracle vector search using COSINE similarity.
        Returns top-k chunks and normalized similarity scores in [0, 1].
        """
        cursor = self.connection.cursor()

        # Ensure query vector is L2-normalized
        array_type = "d" if self.vector_datatype == "FLOAT64" else "f"
        query_vector = array(array_type, normalize_embedding(np.array(embedding)))

        query = f"""
            SELECT
                *
            FROM (
                SELECT
                    content,
                    chunk_id,
                    metadata,
                    chunk_metadata,
                    vector,
                    VECTOR_DISTANCE(vector, :query_vector, COSINE) AS score
                FROM {self.table_name}
            )
        """

        params: dict = {
            "query_vector": query_vector,
        }

        if score_threshold is not None:
            query += " WHERE score >= :score_threshold"
            params["score_threshold"] = score_threshold

        query += " ORDER BY score DESC FETCH FIRST :k ROWS ONLY"
        params["k"] = k

        logger.debug("Executing query", query=query)
        logger.debug("Query vector", vector=str(query_vector))
        try:
            cursor.execute(query, params)
            results = cursor.fetchall()

            chunks: list[EmbeddedChunk] = []
            scores: list[float] = []

            for row in results:
                content, chunk_id, metadata, chunk_metadata, vector, score = row

                chunk = EmbeddedChunk(
                    content=content.read(),
                    chunk_id=chunk_id,
                    metadata=metadata,
                    embedding=vector,
                    chunk_metadata=chunk_metadata,
                    embedding_model=self.vector_store.embedding_model,
                    embedding_dimension=self.vector_store.embedding_dimension,
                )

                chunks.append(chunk)
                scores.append(float(score))
            logger.debug("result count", chunks_count=len(chunks))
            return QueryChunksResponse(chunks=chunks, scores=scores)

        except Exception as e:
            logger.error("Failed to query vector", error=str(e))
            raise

        finally:
            cursor.close()

    async def query_keyword(
        self, query_string: str, k: int, score_threshold: float, filters: Filter | None = None
    ) -> QueryChunksResponse:
        cursor = self.connection.cursor()

        # Build base query
        base_query = f"""
                SELECT
                    content,
                    chunk_id,
                    metadata,
                    chunk_metadata,
                    vector,
                    score / max_score AS score
                FROM (
                    SELECT
                        content,
                        chunk_id,
                        metadata,
                        chunk_metadata,
                        vector,
                        SCORE(1) AS score,
                        MAX(SCORE(1)) OVER () AS max_score
                    FROM {self.table_name}
                    WHERE CONTAINS(content, :query_string, 1) > 0
                )
        """

        params = {"query_string": query_string, "k": k}

        if score_threshold is not None:
            base_query += " WHERE score >= :score_threshold"
            params["score_threshold"] = score_threshold

        query = base_query + " ORDER BY score DESC FETCH FIRST :k ROWS ONLY;"

        logger.debug(query)

        try:
            cursor.execute(query, params)
            results = cursor.fetchall()

            chunks = []
            scores = []
            for row in results:
                content, chunk_id, metadata, chunk_metadata, vector, score = row
                chunk = EmbeddedChunk(
                    content=content.read(),
                    chunk_id=chunk_id,
                    metadata=metadata,
                    embedding=vector,
                    chunk_metadata=chunk_metadata,
                    embedding_model=self.vector_store.embedding_model,
                    embedding_dimension=self.vector_store.embedding_dimension,
                )
                chunks.append(chunk)
                scores.append(float(score))
            logger.debug("result count", chunks_count=len(chunks))
            return QueryChunksResponse(chunks=chunks, scores=scores)
        except Exception as e:
            logger.error("Error performing keyword search", error=str(e))
            raise
        finally:
            cursor.close()

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

        # Get results from both search methods
        vector_response = await self.query_vector(embedding, k, score_threshold)
        keyword_response = await self.query_keyword(query_string, k, score_threshold)

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
        filtered_items = [(doc_id, score) for doc_id, score in top_k_items if score >= (score_threshold or 0)]

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
        try:
            with self.connection.cursor() as cursor:
                cursor.execute(f"DROP TABLE IF EXISTS {self.table_name}")
            logger.info("Dropped table: {self.table_name}")
        except oracledb.DatabaseError as e:
            logger.error("Error dropping table", table_name=self.table_name, error=str(e))
            raise

    async def delete_chunks(self, chunks_for_deletion: list[ChunkForDeletion]) -> None:
        chunk_ids = [c.chunk_id for c in chunks_for_deletion]
        if not chunk_ids:
            return
        placeholders = [f":id_{i}" for i in range(len(chunk_ids))]
        params = {f"id_{i}": cid for i, cid in enumerate(chunk_ids)}
        cursor = self.connection.cursor()
        try:
            cursor.execute(
                f"DELETE FROM {self.table_name} WHERE chunk_id IN ({', '.join(placeholders)})",
                params,
            )
        except Exception as e:
            logger.error("Error deleting chunks from Oracle 26AI table", table_name=self.table_name, error=str(e))
            raise
        finally:
            cursor.close()


class OCI26aiVectorIOAdapter(OpenAIVectorStoreMixin, VectorIO, VectorStoresProtocolPrivate):
    """Vector I/O adapter for Oracle 26ai database."""

    def __init__(
        self,
        config: OCI26aiVectorIOConfig,
        inference_api: Inference,
        files_api: Files | None,
        file_processor_api: FileProcessors | None = None,
    ) -> None:
        super().__init__(
            inference_api=inference_api, files_api=files_api, kvstore=None, file_processor_api=file_processor_api
        )
        self.config = config
        self.cache: dict[str, VectorStoreWithIndex] = {}
        self.pool = None
        self.inference_api = inference_api
        self.vector_store_table = None

    async def initialize(self) -> None:
        logger.info("Initializing OCI26aiVectorIOAdapter")
        self.kvstore = await kvstore_impl(self.config.persistence)
        await self.initialize_openai_vector_stores()

        try:
            self.connection = oracledb.connect(
                user=self.config.user,
                password=self.config.password,
                dsn=self.config.conn_str,
                config_dir=self.config.tnsnames_loc,
                wallet_location=self.config.ewallet_pem_loc,
                wallet_password=self.config.ewallet_password,
                expire_time=1,  # minutes
            )
            self.connection.autocommit = True
            logger.info("Oracle connection created successfully")
        except Exception as e:
            logger.error("Error creating Oracle connection", error=str(e))
            raise

        # Load State
        start_key = OPENAI_VECTOR_STORES_PREFIX
        end_key = f"{OPENAI_VECTOR_STORES_PREFIX}\xff"
        stored_vector_stores = await self.kvstore.values_in_range(start_key, end_key)
        for vector_store_data in stored_vector_stores:
            vector_store = VectorStore.model_validate_json(vector_store_data)
            logger.info(
                "Loading index",
                vector_store_name=vector_store.vector_store_name,
                vector_store_id=vector_store.vector_store_id,
            )
            oci_index = OCI26aiIndex(
                connection=self.connection,
                vector_store=vector_store,
                kvstore=self.kvstore,
                vector_datatype=self.config.vector_datatype,
            )
            await oci_index.initialize()
            index = VectorStoreWithIndex(vector_store, index=oci_index, inference_api=self.inference_api)
            self.cache[vector_store.identifier] = index

        logger.info("Completed loading indexes", stored_vector_stores_count=len(stored_vector_stores))

    async def shutdown(self) -> None:
        logger.info("Shutting down Oracle connection")
        if self.connection is not None:
            self.connection.close()
        # Clean up mixin resources (file batch tasks)
        await super().shutdown()

    async def register_vector_store(self, vector_store: VectorStore) -> None:
        if self.kvstore is None:
            raise RuntimeError("KVStore not initialized. Call initialize() before registering vector stores.")

        # # Save to kvstore for persistence
        key = f"{OPENAI_VECTOR_STORES_PREFIX}{vector_store.identifier}"
        await self.kvstore.set(key=key, value=vector_store.model_dump_json())

        if isinstance(self.config, OCI26aiVectorIOConfig):
            consistency_level = self.config.consistency_level
        else:
            consistency_level = "Strong"
        oci_index = OCI26aiIndex(
            connection=self.connection,
            vector_store=vector_store,
            consistency_level=consistency_level,
            vector_datatype=self.config.vector_datatype,
        )
        index = VectorStoreWithIndex(
            vector_store=vector_store,
            index=oci_index,
            inference_api=self.inference_api,
        )
        await oci_index.initialize()
        self.cache[vector_store.identifier] = index

    async def _get_and_cache_vector_store_index(self, vector_store_id: str) -> VectorStoreWithIndex | None:
        if vector_store_id in self.cache:
            return self.cache[vector_store_id]

        # Try to load from kvstore
        if self.kvstore is None:
            raise RuntimeError("KVStore not initialized. Call initialize() before using vector stores.")

        key = f"{OPENAI_VECTOR_STORES_PREFIX}{vector_store_id}"
        vector_store_data = await self.kvstore.get(key)
        if not vector_store_data:
            raise VectorStoreNotFoundError(vector_store_id)

        vector_store = VectorStore.model_validate_json(vector_store_data)
        index = VectorStoreWithIndex(
            vector_store=vector_store,
            index=OCI26aiIndex(
                connection=self.connection,
                vector_store=vector_store,
                kvstore=self.kvstore,
                vector_datatype=self.config.vector_datatype,
            ),
            inference_api=self.inference_api,
        )
        self.cache[vector_store_id] = index
        return index

    async def unregister_vector_store(self, vector_store_id: str) -> None:
        # Remove provider index and cache
        if vector_store_id in self.cache:
            await self.cache[vector_store_id].index.delete()
            del self.cache[vector_store_id]

        # Delete vector DB metadata from KV store
        if self.kvstore is None:
            raise RuntimeError("KVStore not initialized. Call initialize() before unregistering vector stores.")
        await self.kvstore.delete(key=f"{OPENAI_VECTOR_STORES_PREFIX}{vector_store_id}")

    async def insert_chunks(self, request: InsertChunksRequest) -> None:
        index = await self._get_and_cache_vector_store_index(request.vector_store_id)
        if not index:
            raise VectorStoreNotFoundError(request.vector_store_id)

        await index.insert_chunks(request)

    async def query_chunks(self, request: QueryChunksRequest) -> QueryChunksResponse:
        index = await self._get_and_cache_vector_store_index(request.vector_store_id)
        if not index:
            raise VectorStoreNotFoundError(request.vector_store_id)

        # Ensure embedding_dimensions is set in params
        params = request.params.copy() if request.params else {}
        if "embedding_dimensions" not in params:
            params["embedding_dimensions"] = index.vector_store.embedding_dimension

        # Create updated request with modified params
        updated_request = request.model_copy(update={"params": params})
        return await index.query_chunks(updated_request)

    async def delete_chunks(self, request: DeleteChunksRequest) -> None:
        """Delete chunks from an OCI 26AI vector store."""
        index = await self._get_and_cache_vector_store_index(request.vector_store_id)
        if not index:
            raise VectorStoreNotFoundError(request.vector_store_id)

        await index.index.delete_chunks(request.chunks)
