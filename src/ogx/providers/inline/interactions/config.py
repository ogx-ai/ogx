# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

from pydantic import BaseModel, Field

from ogx.core.storage.datatypes import SqlStoreReference


class InteractionsConfig(BaseModel):
    """Configuration for the built-in Google Interactions API adapter."""

    store: SqlStoreReference = Field(
        default_factory=lambda: SqlStoreReference(
            backend="sql_default",
            table_name="interactions",
        ),
        description="SQL store for persisting interaction state (conversation chaining).",
    )

    @classmethod
    def sample_run_config(cls, __distro_dir__: str = "") -> dict[str, Any]:
        return {
            "store": SqlStoreReference(
                backend="sql_default",
                table_name="interactions",
            ).model_dump(exclude_none=True),
        }
