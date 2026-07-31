# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

from pydantic import BaseModel, Field

from ogx.core.storage.datatypes import KVStoreReference, SqlStoreReference
from ogx_api import json_schema_type


@json_schema_type
class OpenSearchVectorIOConfig(BaseModel):
    """Configuration for the OpenSearch vector I/O provider."""

    host: str = Field(default="localhost", description="The host of the OpenSearch server")
    port: int = Field(default=9200, description="The port of the OpenSearch server")
    use_ssl: bool = Field(default=False, description="Whether to use SSL for the connection")
    verify_certs: bool = Field(default=False, description="Whether to verify SSL certificates")
    username: str | None = Field(default=None, description="The username for authentication")
    password: str | None = Field(default=None, description="The password for authentication")
    persistence: KVStoreReference | None = Field(
        description="Config for KV store backend (SQLite only for now)", default=None
    )
    metadata_store: SqlStoreReference | None = Field(
        default=None,
        description="SQL store reference for tenant-isolated vector store metadata",
    )

    @classmethod
    def sample_run_config(cls, __distro_dir__: str, **kwargs: Any) -> dict[str, Any]:
        return {
            "host": "${env.OPENSEARCH_HOST:=localhost}",
            "port": 9200,
            "use_ssl": False,
            "verify_certs": False,
            "username": "${env.OPENSEARCH_USERNAME:=}",
            "password": "${env.OPENSEARCH_PASSWORD:=}",
            "persistence": KVStoreReference(
                backend="kv_default",
                namespace="vector_io::opensearch",
            ).model_dump(exclude_none=True),
        }
