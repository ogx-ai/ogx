# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import asyncio
import hashlib
import heapq
import json
import re
from collections.abc import Sequence
from typing import Any, cast

from neo4j import AsyncDriver, AsyncGraphDatabase
from numpy.typing import NDArray

from ogx.core.access_control.datatypes import AccessRule
from ogx.core.storage.kvstore import kvstore_impl
from ogx.core.storage.sqlstore import authorized_sqlstore
from ogx.log import get_logger
from ogx.providers.utils.inference.prompt_adapter import interleaved_content_as_str
from ogx.providers.utils.memory.openai_vector_store_mixin import OpenAIVectorStoreMixin
from ogx.providers.utils.memory.vector_store import ChunkForDeletion, EmbeddingIndex, VectorStoreWithIndex
from ogx.providers.utils.vector_io.filters import Filter, parse_filter
from ogx.providers.utils.vector_io.vector_utils import (
    WeightedInMemoryAggregator,
    load_embedded_chunk_with_backward_compat,
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

from .config import Neo4jVectorIOConfig

logger = get_logger(name=__name__, category="vector_io::neo4j")

VERSION = "v1"
VECTOR_DBS_PREFIX = f"vector_stores:neo4j:{VERSION}::"
_VALID_IDENTIFIER = re.compile(r"^[A-Za-z0-9_]+$")
_VALID_METADATA_KEY = re.compile(r"^[A-Za-z0-9_-]+$")
_CYPHER_OPS = {
    "eq": "=",
    "ne": "<>",
    "gt": ">",
    "gte": ">=",
    "lt": "<",
    "lte": "<=",
}
_FILTER_CANDIDATE_MULTIPLIER = 10
_MAX_GRAPH_NEIGHBORS = 1000


def _sanitize_identifier(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_]", "_", value)
    if not sanitized or sanitized[0].isdigit():
        sanitized = f"_{sanitized}"
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]
    return f"{sanitized}_{digest}"


def _quote_identifier(value: str) -> str:
    if not _VALID_IDENTIFIER.match(value):
        raise ValueError(f"Failed to quote Neo4j identifier: invalid identifier {value!r}")
    return f"`{value}`"


def _metadata_property(key: str) -> str:
    if not _VALID_METADATA_KEY.match(key):
        raise ValueError(f"Failed to translate Neo4j metadata filter: invalid metadata key {key!r}")
    return f"metadata_{key}"


def _quote_property(value: str) -> str:
    return f"`{value.replace('`', '``')}`"


def _is_neo4j_property_value(value: Any) -> bool:
    if isinstance(value, str | int | float | bool) or value is None:
        return True
    if isinstance(value, list):
        return all(isinstance(item, str | int | float | bool) or item is None for item in value)
    return False


def _metadata_properties(metadata: dict[str, Any]) -> dict[str, Any]:
    properties: dict[str, Any] = {}
    for key, value in metadata.items():
        if not _VALID_METADATA_KEY.match(key):
            raise ValueError(f"Failed to prepare Neo4j metadata: invalid metadata key {key!r}")
        if not _is_neo4j_property_value(value):
            raise ValueError(f"Failed to prepare Neo4j metadata: value for key {key!r} is not a Neo4j property value")
        properties[_metadata_property(key)] = value
    return properties


def _load_chunk(chunk_data: Any) -> EmbeddedChunk:
    if isinstance(chunk_data, str):
        chunk_data = json.loads(chunk_data)
    return load_embedded_chunk_with_backward_compat(chunk_data)


def _translate_filters(filters: Filter | dict[str, Any] | None) -> tuple[str, dict[str, Any]]:
    if filters is None:
        return "", {}

    parsed = parse_filter(filters)
    counter = 0

    def translate(filter_obj: ComparisonFilter | CompoundFilter) -> tuple[str, dict[str, Any]]:
        nonlocal counter
        if isinstance(filter_obj, ComparisonFilter):
            property_name = _metadata_property(filter_obj.key)
            param_name = f"filter_{counter}"
            counter += 1
            if filter_obj.type in _CYPHER_OPS:
                return f"node.{_quote_property(property_name)} {_CYPHER_OPS[filter_obj.type]} ${param_name}", {
                    param_name: filter_obj.value
                }
            if filter_obj.type == "in":
                return f"node.{_quote_property(property_name)} IN ${param_name}", {param_name: filter_obj.value}
            if filter_obj.type == "nin":
                return f"NOT node.{_quote_property(property_name)} IN ${param_name}", {param_name: filter_obj.value}
            raise ValueError(f"Failed to translate Neo4j metadata filter: unsupported type {filter_obj.type}")

        joiner = " AND " if filter_obj.type == "and" else " OR "
        clauses = []
        params: dict[str, Any] = {}
        for sub_filter in filter_obj.filters:
            sub_parsed = parse_filter(sub_filter)
            clause, sub_params = translate(sub_parsed)
            clauses.append(f"({clause})")
            params.update(sub_params)
        return joiner.join(clauses), params

    return translate(parsed)


def _candidate_limit(k: int, filters: Filter | None) -> int:
    if filters is None:
        return k
    return k * _FILTER_CANDIDATE_MULTIPLIER


def _validate_graph_int(name: str, value: Any, minimum: int, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"Failed to expand Neo4j graph: {name} must be an integer")
    if value < minimum or (maximum is not None and value > maximum):
        bounds = f"between {minimum} and {maximum}" if maximum is not None else f"at least {minimum}"
        raise ValueError(f"Failed to expand Neo4j graph: {name} must be {bounds}")
    return cast(int, value)


def _validate_graph_weight(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError("Failed to expand Neo4j graph: graph_expansion_weight must be a number")
    weight = float(value)
    if not 0.0 <= weight <= 1.0:
        raise ValueError("Failed to expand Neo4j graph: graph_expansion_weight must be between 0 and 1")
    return weight


def _validate_relationship_types(value: Any) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, str) or not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError("Failed to expand Neo4j graph: graph_relationship_types must be a list of strings")
    for relationship_type in value:
        _quote_identifier(relationship_type)
    return cast(list[str], value)


class Neo4jIndex(EmbeddingIndex):
    """Neo4j-backed embedding index for one OGX vector store."""

    def __init__(self, driver: AsyncDriver, config: Neo4jVectorIOConfig, vector_store: VectorStore) -> None:
        self.driver = driver
        self.config = config
        self.vector_store = vector_store
        self.database = config.database
        self.dimension = vector_store.embedding_dimension
        store_id = _sanitize_identifier(vector_store.identifier)
        prefix = _sanitize_identifier(config.index_prefix)
        self.chunk_label = f"OGXChunk_{store_id}"
        self.chunk_constraint_name = f"{prefix}_{store_id}_chunk_id"
        self.vector_index_name = f"{prefix}_{store_id}_embedding"
        self.fulltext_index_name = f"{prefix}_{store_id}_content"

    async def initialize(self) -> None:
        chunk_label = _quote_identifier(self.chunk_label)
        chunk_constraint_name = _quote_identifier(self.chunk_constraint_name)
        vector_index_name = _quote_identifier(self.vector_index_name)
        fulltext_index_name = _quote_identifier(self.fulltext_index_name)
        async with self.driver.session(database=self.database) as session:
            await session.run(
                f"""
                CREATE CONSTRAINT {chunk_constraint_name} IF NOT EXISTS
                FOR (c:{chunk_label}) REQUIRE c.chunk_id IS UNIQUE
                """
            )
            await session.run(
                f"""
                CREATE VECTOR INDEX {vector_index_name} IF NOT EXISTS
                FOR (c:{chunk_label}) ON (c.embedding)
                OPTIONS {{indexConfig: {{`vector.dimensions`: $dimension, `vector.similarity_function`: 'cosine'}}}}
                """,
                dimension=self.dimension,
            )
            await session.run(
                f"""
                CREATE FULLTEXT INDEX {fulltext_index_name} IF NOT EXISTS
                FOR (c:{chunk_label}) ON EACH [c.content]
                """
            )
            await session.run("CALL db.awaitIndexes($seconds)", seconds=30)

    async def add_chunks(self, embedded_chunks: list[EmbeddedChunk]) -> None:
        if not embedded_chunks:
            return

        rows = []
        for chunk in embedded_chunks:
            rows.append(
                {
                    "chunk_id": chunk.chunk_id,
                    "content": interleaved_content_as_str(chunk.content),
                    "embedding": chunk.embedding,
                    "chunk_content": json.dumps(chunk.model_dump(mode="json")),
                    "metadata_properties": _metadata_properties(chunk.metadata),
                }
            )

        async with self.driver.session(database=self.database) as session:
            await session.run(
                f"""
                UNWIND $rows AS row
                MERGE (node:{_quote_identifier(self.chunk_label)} {{chunk_id: row.chunk_id}})
                SET node = {{}}
                SET node.chunk_id = row.chunk_id,
                    node.content = row.content,
                    node.embedding = row.embedding,
                    node.chunk_content = row.chunk_content,
                    node.vector_store_id = $vector_store_id
                SET node += row.metadata_properties
                """,
                rows=rows,
                vector_store_id=self.vector_store.identifier,
            )

    async def delete_chunks(self, chunks_for_deletion: list[ChunkForDeletion]) -> None:
        chunk_ids = [chunk.chunk_id for chunk in chunks_for_deletion]
        if not chunk_ids:
            return

        async with self.driver.session(database=self.database) as session:
            await session.run(
                f"""
                MATCH (node:{_quote_identifier(self.chunk_label)})
                WHERE node.chunk_id IN $chunk_ids
                DETACH DELETE node
                """,
                chunk_ids=chunk_ids,
            )

    async def query_vector(
        self, embedding: NDArray, k: int, score_threshold: float, filters: Filter | None = None
    ) -> QueryChunksResponse:
        filter_clause, filter_params = _translate_filters(filters)
        filter_cypher = f" AND {filter_clause}" if filter_clause else ""
        candidate_limit = _candidate_limit(k, filters)
        async with self.driver.session(database=self.database) as session:
            result = await session.run(
                f"""
                CALL db.index.vector.queryNodes($index_name, $candidate_limit, $embedding)
                YIELD node, score
                WHERE score >= $score_threshold{filter_cypher}
                RETURN node.chunk_content AS chunk_content, score
                ORDER BY score DESC
                LIMIT $limit
                """,
                index_name=self.vector_index_name,
                candidate_limit=candidate_limit,
                limit=k,
                embedding=embedding.tolist(),
                score_threshold=score_threshold,
                **filter_params,
            )
            rows = await result.data()

        return QueryChunksResponse(
            chunks=[_load_chunk(row["chunk_content"]) for row in rows],
            scores=[float(row["score"]) for row in rows],
        )

    async def query_keyword(
        self, query_string: str, k: int, score_threshold: float, filters: Filter | None = None
    ) -> QueryChunksResponse:
        filter_clause, filter_params = _translate_filters(filters)
        filter_cypher = f" AND {filter_clause}" if filter_clause else ""
        candidate_limit = _candidate_limit(k, filters)
        async with self.driver.session(database=self.database) as session:
            result = await session.run(
                f"""
                CALL db.index.fulltext.queryNodes($index_name, $search_query, {{limit: $candidate_limit}})
                YIELD node, score
                WHERE score >= $score_threshold{filter_cypher}
                RETURN node.chunk_content AS chunk_content, score
                ORDER BY score DESC
                LIMIT $limit
                """,
                index_name=self.fulltext_index_name,
                search_query=query_string,
                candidate_limit=candidate_limit,
                limit=k,
                score_threshold=score_threshold,
                **filter_params,
            )
            rows = await result.data()

        return QueryChunksResponse(
            chunks=[_load_chunk(row["chunk_content"]) for row in rows],
            scores=[float(row["score"]) for row in rows],
        )

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
            vector_scores,
            keyword_scores,
            reranker_type,
            reranker_params,
        )
        chunk_map = {chunk.chunk_id: chunk for chunk in vector_response.chunks + keyword_response.chunks}
        ranked = heapq.nlargest(k, combined_scores.items(), key=lambda item: item[1])
        chunks = [
            chunk_map[chunk_id] for chunk_id, score in ranked if score >= score_threshold and chunk_id in chunk_map
        ]
        scores = [score for chunk_id, score in ranked if score >= score_threshold and chunk_id in chunk_map]
        return QueryChunksResponse(chunks=chunks, scores=scores)

    async def expand_graph(
        self,
        response: QueryChunksResponse,
        k: int,
        params: dict[str, Any] | None = None,
    ) -> QueryChunksResponse:
        params = params or {}
        enabled = bool(params.get("graph_retrieval_enabled", self.config.graph_retrieval_enabled))
        if not enabled or not response.chunks:
            return response

        seed_ids = [chunk.chunk_id for chunk in response.chunks]
        max_neighbors = _validate_graph_int(
            "graph_max_neighbors",
            params.get("graph_max_neighbors", self.config.graph_max_neighbors),
            1,
            _MAX_GRAPH_NEIGHBORS,
        )
        weight = _validate_graph_weight(params.get("graph_expansion_weight", self.config.graph_expansion_weight))
        depth = _validate_graph_int(
            "graph_expansion_depth",
            params.get("graph_expansion_depth", self.config.graph_expansion_depth),
            1,
            3,
        )
        relationship_types = _validate_relationship_types(
            params.get("graph_relationship_types", self.config.graph_relationship_types)
        )

        existing_scores = {
            chunk.chunk_id: score for chunk, score in zip(response.chunks, response.scores, strict=False)
        }
        chunk_map = {chunk.chunk_id: chunk for chunk in response.chunks}
        neighbors = await self._query_graph_neighbors(seed_ids, max_neighbors, depth, relationship_types)

        for chunk_id, chunk, related_seed_ids in neighbors:
            seed_scores = [existing_scores[seed_id] for seed_id in related_seed_ids if seed_id in existing_scores]
            if not seed_scores:
                continue
            graph_score = max(seed_scores) * weight
            if graph_score > existing_scores.get(chunk_id, float("-inf")):
                existing_scores[chunk_id] = graph_score
                chunk_map[chunk_id] = chunk

        ranked = heapq.nlargest(k, existing_scores.items(), key=lambda item: item[1])
        return QueryChunksResponse(
            chunks=[chunk_map[chunk_id] for chunk_id, _ in ranked],
            scores=[score for _, score in ranked],
        )

    async def _query_graph_neighbors(
        self,
        seed_ids: Sequence[str],
        max_neighbors: int,
        depth: int,
        relationship_types: list[str] | None,
    ) -> list[tuple[str, EmbeddedChunk, list[str]]]:
        rel_pattern = self._relationship_pattern(depth, relationship_types)
        async with self.driver.session(database=self.database) as session:
            result = await session.run(
                f"""
                MATCH (seed:{_quote_identifier(self.chunk_label)})
                WHERE seed.chunk_id IN $seed_ids
                MATCH (seed){rel_pattern}(neighbor:{_quote_identifier(self.chunk_label)})
                WHERE NOT neighbor.chunk_id IN $seed_ids
                RETURN neighbor.chunk_id AS chunk_id,
                       neighbor.chunk_content AS chunk_content,
                       collect(DISTINCT seed.chunk_id) AS seed_ids
                LIMIT $limit
                """,
                seed_ids=list(seed_ids),
                limit=max_neighbors,
            )
            rows = await result.data()

        return [
            (row["chunk_id"], _load_chunk(row["chunk_content"]), list(row["seed_ids"]))
            for row in rows
            if row.get("chunk_content")
        ]

    def _relationship_pattern(self, depth: int, relationship_types: list[str] | None) -> str:
        safe_depth = max(1, min(depth, 3))
        validated_relationship_types = _validate_relationship_types(relationship_types)
        if validated_relationship_types:
            rel_types = "|".join(_quote_identifier(rel_type) for rel_type in validated_relationship_types)
            return f"-[:{rel_types}*1..{safe_depth}]-"
        return f"-[*1..{safe_depth}]-"

    async def delete(self) -> None:
        async with self.driver.session(database=self.database) as session:
            await session.run(f"MATCH (node:{_quote_identifier(self.chunk_label)}) DETACH DELETE node")
            await session.run(f"DROP CONSTRAINT {_quote_identifier(self.chunk_constraint_name)} IF EXISTS")
            await session.run(f"DROP INDEX {_quote_identifier(self.vector_index_name)} IF EXISTS")
            await session.run(f"DROP INDEX {_quote_identifier(self.fulltext_index_name)} IF EXISTS")


class Neo4jVectorIOAdapter(OpenAIVectorStoreMixin, VectorIO, VectorStoresProtocolPrivate):
    """VectorIO adapter that stores chunks in Neo4j and supports graph expansion."""

    def __init__(
        self,
        config: Neo4jVectorIOConfig,
        inference_api: Inference,
        files_api: Files | None = None,
        file_processor_api: FileProcessors | None = None,
        policy: list[AccessRule] | None = None,
    ) -> None:
        super().__init__(
            inference_api=inference_api,
            files_api=files_api,
            kvstore=None,
            file_processor_api=file_processor_api,
        )
        self.config = config
        self.driver: AsyncDriver | None = None
        self.cache: dict[str, VectorStoreWithIndex] = {}
        self._policy = policy or []
        self._driver_lock = asyncio.Lock()

    async def _ensure_driver(self) -> AsyncDriver:
        async with self._driver_lock:
            if self.driver is not None:
                existing_driver = self.driver
                try:
                    await existing_driver.verify_connectivity()
                    return existing_driver
                except Exception as error:
                    logger.warning("Neo4j driver connectivity check failed; recreating driver", error=str(error))
                    self.driver = None
                    try:
                        await existing_driver.close()
                    except Exception:
                        logger.exception("Failed to close stale Neo4j driver")
            auth = None
            if self.config.user:
                auth = (
                    self.config.user,
                    self.config.password.get_secret_value() if self.config.password else "",
                )
            new_driver = AsyncGraphDatabase.driver(self.config.uri, auth=auth)
            self.driver = new_driver
            try:
                await new_driver.verify_connectivity()
            except Exception:
                self.driver = None
                try:
                    await new_driver.close()
                except Exception:
                    logger.exception("Failed to close Neo4j driver after connectivity failure")
                raise
            return new_driver

    async def initialize(self) -> None:
        logger.info("Initializing Neo4j vector_io adapter", uri=self.config.uri, database=self.config.database)
        self.kvstore = await kvstore_impl(self.config.persistence)
        if self.config.metadata_store:
            self.metadata_store = await authorized_sqlstore(self.config.metadata_store, self._policy)
        await self.initialize_openai_vector_stores()
        driver = await self._ensure_driver()
        try:
            stored_vector_stores = await self.kvstore.values_in_range(VECTOR_DBS_PREFIX, f"{VECTOR_DBS_PREFIX}\xff")
            for vector_store_data in stored_vector_stores:
                vector_store = VectorStore.model_validate_json(vector_store_data)
                neo4j_index = Neo4jIndex(driver, self.config, vector_store)
                await neo4j_index.initialize()
                self.cache[vector_store.identifier] = VectorStoreWithIndex(
                    vector_store,
                    neo4j_index,
                    self.inference_api,
                )
        except Exception:
            await self.shutdown()
            raise

    async def shutdown(self) -> None:
        if self.driver is not None:
            try:
                await self.driver.close()
                logger.info("Closed Neo4j driver")
            except Exception:
                logger.exception("Failed to close Neo4j driver")
            finally:
                self.driver = None
        await super().shutdown()

    async def register_vector_store(self, vector_store: VectorStore) -> None:
        if self.kvstore is None:
            raise RuntimeError("Failed to register Neo4j vector store: KVStore is not initialized")
        driver = await self._ensure_driver()
        neo4j_index = Neo4jIndex(driver, self.config, vector_store)
        await neo4j_index.initialize()
        await self.kvstore.set(
            key=f"{VECTOR_DBS_PREFIX}{vector_store.identifier}",
            value=vector_store.model_dump_json(),
        )
        self.cache[vector_store.identifier] = VectorStoreWithIndex(vector_store, neo4j_index, self.inference_api)

    async def unregister_vector_store(self, vector_store_id: str) -> None:
        if vector_store_id in self.cache:
            await self.cache[vector_store_id].index.delete()
            del self.cache[vector_store_id]
        if self.kvstore is None:
            raise RuntimeError("Failed to unregister Neo4j vector store: KVStore is not initialized")
        await self.kvstore.delete(key=f"{VECTOR_DBS_PREFIX}{vector_store_id}")

    async def _get_and_cache_vector_store_index(self, vector_store_id: str) -> VectorStoreWithIndex:
        if vector_store_id in self.cache:
            return self.cache[vector_store_id]
        if self.kvstore is None:
            raise RuntimeError("Failed to load Neo4j vector store: KVStore is not initialized")
        vector_store_data = await self.kvstore.get(f"{VECTOR_DBS_PREFIX}{vector_store_id}")
        if not vector_store_data:
            raise VectorStoreNotFoundError(vector_store_id)
        vector_store = VectorStore.model_validate_json(vector_store_data)
        neo4j_index = Neo4jIndex(await self._ensure_driver(), self.config, vector_store)
        await neo4j_index.initialize()
        self.cache[vector_store_id] = VectorStoreWithIndex(vector_store, neo4j_index, self.inference_api)
        return self.cache[vector_store_id]

    async def insert_chunks(self, request: InsertChunksRequest) -> None:
        index = await self._get_and_cache_vector_store_index(request.vector_store_id)
        await index.insert_chunks(request)

    async def query_chunks(self, request: QueryChunksRequest) -> QueryChunksResponse:
        index = await self._get_and_cache_vector_store_index(request.vector_store_id)
        response = await index.query_chunks(request)
        if isinstance(index.index, Neo4jIndex):
            params = request.params or {}
            k = params.get("max_chunks", 3)
            return await index.index.expand_graph(response, k, params)
        return response

    async def delete_chunks(self, request: DeleteChunksRequest) -> None:
        index = await self._get_and_cache_vector_store_index(request.vector_store_id)
        await index.index.delete_chunks(request.chunks)
