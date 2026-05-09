# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

from pydantic import BaseModel, Field

from ogx.core.storage.datatypes import KVStoreReference
from ogx_api import json_schema_type


@json_schema_type
class WeaviateVectorIOConfig(BaseModel):
    """Configuration for the Weaviate vector I/O provider."""

    weaviate_api_key: str | None = Field(description="The API key for the Weaviate instance", default=None)
    weaviate_cluster_url: str | None = Field(description="The URL of the Weaviate cluster", default="localhost:8080")
    persistence: KVStoreReference | None = Field(
        description="Config for KV store backend (SQLite only for now)", default=None
    )

    @classmethod
    def sample_run_config(cls, __distro_dir__: str, **kwargs: Any) -> dict[str, Any]:
        return {
            "weaviate_api_key": None,
            "weaviate_cluster_url": "${env.WEAVIATE_CLUSTER_URL:=localhost:8080}",
            "persistence": KVStoreReference(
                backend="kv_default",
                namespace="vector_io::weaviate",
            ).model_dump(exclude_none=True),
        }
