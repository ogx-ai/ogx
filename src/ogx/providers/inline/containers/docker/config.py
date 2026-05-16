# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any, Literal

from pydantic import BaseModel, Field

from ogx.core.storage.datatypes import SqlStoreReference


class DockerContainersImplConfig(BaseModel):
    """Configuration for the Docker containers provider."""

    default_image: str = Field(
        default="python:3.12-slim",
        description="Default container image to use when not specified",
    )
    default_memory_limit: Literal["1g", "4g", "16g", "64g"] = Field(
        default="1g",
        description="Default memory limit for containers",
    )
    metadata_store: SqlStoreReference = Field(
        description="SQL store configuration for container metadata",
    )

    @classmethod
    def sample_run_config(cls, __distro_dir__: str) -> dict[str, Any]:
        return {
            "default_image": "python:3.12-slim",
            "default_memory_limit": "1g",
            "metadata_store": SqlStoreReference(
                backend="sql_default",
                table_name="containers_metadata",
            ).model_dump(exclude_none=True),
        }
