# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

from pydantic import BaseModel, Field, SecretStr

from ogx.core.storage.datatypes import KVStoreReference, SqlStoreReference
from ogx_api import json_schema_type


@json_schema_type
class Neo4jVectorIOConfig(BaseModel):
    """Configuration for the remote Neo4j vector I/O provider."""

    uri: str = Field(default="bolt://localhost:7687", description="Neo4j Bolt URI.")
    user: str = Field(default="neo4j", description="Neo4j username.")
    password: SecretStr | None = Field(default=None, description="Neo4j password.")
    database: str = Field(default="neo4j", description="Neo4j database name.")
    index_prefix: str = Field(default="ogx", description="Prefix for Neo4j vector and full-text indexes.")
    graph_retrieval_enabled: bool = Field(
        default=False,
        description="Whether to expand initial retrieval results through graph relationships.",
    )
    graph_expansion_depth: int = Field(
        default=1,
        ge=1,
        le=3,
        description="Maximum relationship traversal depth for graph-aware retrieval.",
    )
    graph_max_neighbors: int = Field(
        default=10,
        ge=1,
        le=1000,
        description="Maximum graph-expanded neighbor chunks to consider per query.",
    )
    graph_expansion_weight: float = Field(
        default=0.15,
        ge=0.0,
        le=1.0,
        description="Score multiplier applied to graph-expanded chunks relative to seed chunk scores.",
    )
    graph_relationship_types: list[str] | None = Field(
        default=None,
        description="Relationship types used for graph expansion. When omitted, all relationships are considered.",
    )
    persistence: KVStoreReference = Field(description="Config for KV store backend.")
    metadata_store: SqlStoreReference | None = Field(
        default=None,
        description="SQL store reference for tenant-isolated vector store metadata.",
    )

    @classmethod
    def sample_run_config(cls, __distro_dir__: str, **kwargs: Any) -> dict[str, Any]:
        return {
            "uri": "${env.NEO4J_URI:=bolt://localhost:7687}",
            "user": "${env.NEO4J_USER:=neo4j}",
            "password": "${env.NEO4J_PASSWORD:=}",
            "database": "${env.NEO4J_DATABASE:=neo4j}",
            "graph_retrieval_enabled": False,
            "graph_expansion_depth": 1,
            "graph_max_neighbors": 10,
            "graph_expansion_weight": 0.15,
            "persistence": KVStoreReference(
                backend="kv_default",
                namespace="vector_io::neo4j",
            ).model_dump(exclude_none=True),
        }
