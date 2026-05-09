# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

from pydantic import BaseModel, Field

from ogx.core.storage.datatypes import SqlStoreReference


class LocalfsFilesImplConfig(BaseModel):
    """Configuration for the local filesystem files provider."""

    storage_dir: str = Field(
        description="Directory to store uploaded files",
    )
    metadata_store: SqlStoreReference = Field(
        description="SQL store configuration for file metadata",
    )
    ttl_secs: int = 365 * 24 * 60 * 60  # 1 year

    @classmethod
    def sample_run_config(cls, __distro_dir__: str) -> dict[str, Any]:
        return {
            "storage_dir": "${env.FILES_STORAGE_DIR:=" + __distro_dir__ + "/files}",
            "metadata_store": SqlStoreReference(
                backend="sql_default",
                table_name="files_metadata",
            ).model_dump(exclude_none=True),
        }
