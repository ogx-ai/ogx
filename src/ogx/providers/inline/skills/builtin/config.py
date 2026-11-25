# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

from pydantic import BaseModel, Field

from ogx.core.storage.datatypes import KVStoreReference, SqlStoreReference


class BuiltinSkillsConfig(BaseModel):
    """Configuration for the built-in skills provider."""

    sqlstore: SqlStoreReference = Field(
        default_factory=lambda: SqlStoreReference(
            backend="sql_default",
            table_name="skills",
        ),
        description="Configuration for the SQL store backend used to persist skill metadata.",
    )

    persistence: KVStoreReference = Field(
        description="KV store reference used to read skill metadata written by pre-authorization "
        "versions during the one-way startup migration",
    )

    @classmethod
    def sample_run_config(cls, __distro_dir__: str) -> dict[str, Any]:
        return {
            "sqlstore": SqlStoreReference(
                backend="sql_default",
                table_name="skills",
            ).model_dump(exclude_none=True),
            "persistence": KVStoreReference(
                backend="kv_default",
                namespace="skills",
            ).model_dump(exclude_none=True),
        }
