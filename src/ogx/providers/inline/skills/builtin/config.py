# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

from pydantic import BaseModel, Field

from ogx.core.storage.datatypes import KVStoreReference


class BuiltinSkillsConfig(BaseModel):
    """Configuration for the built-in skills provider."""

    persistence: KVStoreReference = Field(
        description="KV store reference for skill metadata persistence",
    )

    @classmethod
    def sample_run_config(cls, __distro_dir__: str) -> dict[str, Any]:
        return {
            "persistence": KVStoreReference(
                backend="kv_default",
                namespace="skills",
            ).model_dump(exclude_none=True),
        }
