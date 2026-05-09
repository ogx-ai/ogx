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
class FaissVectorIOConfig(BaseModel):
    """Configuration for the inline FAISS vector I/O provider."""

    persistence: KVStoreReference

    @classmethod
    def sample_run_config(cls, __distro_dir__: str, **kwargs: Any) -> dict[str, Any]:
        return {
            "persistence": KVStoreReference(
                backend="kv_default",
                namespace="vector_io::faiss",
            ).model_dump(exclude_none=True)
        }
