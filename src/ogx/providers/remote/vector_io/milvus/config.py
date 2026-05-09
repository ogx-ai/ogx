# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ogx.core.storage.datatypes import KVStoreReference
from ogx_api import json_schema_type


@json_schema_type
class MilvusVectorIOConfig(BaseModel):
    """Configuration for the remote Milvus vector I/O provider."""

    uri: str = Field(description="The URI of the Milvus server")
    token: str | None = Field(description="The token of the Milvus server")
    consistency_level: str = Field(description="The consistency level of the Milvus server", default="Strong")
    persistence: KVStoreReference = Field(description="Config for KV store backend")

    # This configuration allows additional fields to be passed through to the underlying Milvus client.
    # See the [Milvus](https://milvus.io/docs/install-overview.md) documentation for more details about Milvus in general.
    model_config = ConfigDict(extra="allow")

    @classmethod
    def sample_run_config(cls, __distro_dir__: str, **kwargs: Any) -> dict[str, Any]:
        return {
            "uri": "${env.MILVUS_ENDPOINT}",
            "token": "${env.MILVUS_TOKEN}",
            "persistence": KVStoreReference(
                backend="kv_default",
                namespace="vector_io::milvus_remote",
            ).model_dump(exclude_none=True),
        }
