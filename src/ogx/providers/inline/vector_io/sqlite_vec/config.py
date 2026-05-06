# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

from pydantic import BaseModel, Field

from ogx.core.storage.datatypes import KVStoreReference


class SQLiteVectorIOConfig(BaseModel):
    """Configuration for the SQLite-vec vector I/O provider."""

    db_path: str = Field(description="Path to the SQLite database file")
    persistence: KVStoreReference = Field(description="Config for KV store backend (SQLite only for now)")

    @classmethod
    def sample_run_config(cls, __distro_dir__: str) -> dict[str, Any]:
        return {
            "db_path": "${env.SQLITE_STORE_DIR:=" + __distro_dir__ + "}/" + "sqlite_vec.db",
            "persistence": KVStoreReference(
                backend="kv_default",
                namespace="vector_io::sqlite_vec",
            ).model_dump(exclude_none=True),
        }
