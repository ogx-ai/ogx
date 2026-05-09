# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


from typing import Any

from pydantic import BaseModel

from ogx.core.storage.datatypes import KVStoreReference
from ogx_api import json_schema_type


@json_schema_type
class QdrantVectorIOConfig(BaseModel):
    """Configuration for the inline Qdrant vector I/O provider."""

    path: str
    persistence: KVStoreReference

    @classmethod
    def sample_run_config(cls, __distro_dir__: str) -> dict[str, Any]:
        return {
            "path": "${env.QDRANT_PATH:=~/.ogx/" + __distro_dir__ + "}/" + "qdrant.db",
            "persistence": KVStoreReference(
                backend="kv_default",
                namespace="vector_io::qdrant",
            ).model_dump(exclude_none=True),
        }
