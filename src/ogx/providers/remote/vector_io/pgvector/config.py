# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from enum import StrEnum
from typing import Annotated, Any, Literal, Self

from pydantic import BaseModel, Field, model_validator

from ogx.core.storage.datatypes import KVStoreReference
from ogx_api import json_schema_type


class PGVectorIndexType(StrEnum):
    """Supported pgvector vector index types in OGX."""

    HNSW = "HNSW"
    IVFFlat = "IVFFlat"


class PGVectorHNSWVectorIndex(BaseModel):
    """Configuration for PGVector HNSW (Hierarchical Navigable Small Worlds) vector index.
    https://github.com/pgvector/pgvector?tab=readme-ov-file#hnsw
    """

    type: Literal[PGVectorIndexType.HNSW] = PGVectorIndexType.HNSW
    m: int | None = Field(
        gt=0,
        default=16,
        description="PGVector's HNSW index parameter - maximum number of edges each vertex has to its neighboring vertices in the graph",
    )
    ef_construction: int | None = Field(
        gt=0,
        default=64,
        description="PGVector's HNSW index parameter - size of the dynamic candidate list used for graph construction",
    )
    ef_search: int | None = Field(
        gt=0,
        default=40,
        description="PGVector's HNSW index parameter - a max size of max heap that holds best candidates when traversing the graph",
    )


class PGVectorIVFFlatVectorIndex(BaseModel):
    """Configuration for PGVector IVFFlat (Inverted File with Flat Compression) vector index.
    https://github.com/pgvector/pgvector?tab=readme-ov-file#ivfflat
    """

    type: Literal[PGVectorIndexType.IVFFlat] = PGVectorIndexType.IVFFlat
    lists: int | None = Field(
        gt=0, default=100, description="PGVector's IVFFlat index parameter - number of lists index divides vectors into"
    )
    probes: int | None = Field(
        gt=0,
        default=10,
        description="PGVector's IVFFlat index parameter - number of lists index searches through during ANN search",
    )

    @model_validator(mode="after")
    def validate_probes(self) -> Self:
        if self.probes >= self.lists:
            raise ValueError(
                "probes parameter for PGVector IVFFlat index can't be greater than or equal to the number of lists in the index to allow ANN search."
            )
        return self


PGVectorIndexConfig = Annotated[
    PGVectorHNSWVectorIndex | PGVectorIVFFlatVectorIndex,
    Field(discriminator="type"),
]


@json_schema_type
class PGVectorVectorIOConfig(BaseModel):
    """Configuration for the PGVector vector I/O provider."""

    host: str | None = Field(default="localhost")
    port: int | None = Field(default=5432)
    db: str | None = Field(default="postgres")
    user: str | None = Field(default="postgres")
    password: str | None = Field(default="mysecretpassword")
    distance_metric: Literal["COSINE", "L2", "L1", "INNER_PRODUCT"] | None = Field(
        default="COSINE", description="PGVector distance metric used for vector search in PGVectorIndex"
    )
    vector_index: PGVectorIndexConfig | None = Field(
        default_factory=PGVectorHNSWVectorIndex,
        description="PGVector vector index used for Approximate Nearest Neighbor (ANN) search",
    )
    persistence: KVStoreReference | None = Field(
        description="Config for KV store backend (SQLite only for now)", default=None
    )

    @classmethod
    def sample_run_config(
        cls,
        __distro_dir__: str,
        host: str = "${env.PGVECTOR_HOST:=localhost}",
        port: int = "${env.PGVECTOR_PORT:=5432}",
        db: str = "${env.PGVECTOR_DB}",
        user: str = "${env.PGVECTOR_USER}",
        password: str = "${env.PGVECTOR_PASSWORD}",
        **kwargs: Any,
    ) -> dict[str, Any]:
        return {
            "host": host,
            "port": port,
            "db": db,
            "user": user,
            "password": password,
            "distance_metric": "COSINE",
            "vector_index": PGVectorHNSWVectorIndex(m=16, ef_construction=64, ef_search=40).model_dump(
                mode="json", exclude_none=True
            ),
            "persistence": KVStoreReference(
                backend="kv_default",
                namespace="vector_io::pgvector",
            ).model_dump(exclude_none=True),
        }
