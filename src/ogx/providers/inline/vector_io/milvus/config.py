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
class MilvusVectorIOConfig(BaseModel):
    """Configuration for the inline Milvus vector I/O provider."""

    db_path: str
    persistence: KVStoreReference = Field(description="Config for KV store backend (SQLite only for now)")
    consistency_level: str = Field(description="The consistency level of the Milvus server", default="Strong")

    @classmethod
    def sample_run_config(cls, __distro_dir__: str, **kwargs: Any) -> dict[str, Any]:
        return {
            "db_path": "${env.MILVUS_DB_PATH:=" + __distro_dir__ + "}/" + "milvus.db",
            "persistence": KVStoreReference(
                backend="kv_default",
                namespace="vector_io::milvus",
            ).model_dump(exclude_none=True),
        }
