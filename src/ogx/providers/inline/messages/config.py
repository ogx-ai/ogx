# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

from pydantic import BaseModel, Field

from ogx.core.storage.datatypes import KVStoreReference, SqlStoreReference


class MessagesConfig(BaseModel):
    """Configuration for the built-in Anthropic Messages API adapter."""

    sqlstore: SqlStoreReference = Field(
        default_factory=lambda: SqlStoreReference(
            backend="sql_default",
            table_name="message_batches",
        ),
        description="Configuration for the SQL store backend used to persist message batch metadata.",
    )

    kvstore: KVStoreReference = Field(
        description="Configuration for the key-value store backend used to persist message batch results.",
    )

    max_concurrent_batches: int = Field(
        default=1,
        description="Maximum number of concurrent message batches to process simultaneously.",
        ge=1,
    )

    max_concurrent_requests_per_batch: int = Field(
        default=10,
        description="Maximum number of concurrent requests to process per batch.",
        ge=1,
    )

    @classmethod
    def sample_run_config(cls, __distro_dir__: str = "") -> dict[str, Any]:
        return {
            "sqlstore": SqlStoreReference(
                backend="sql_default",
                table_name="message_batches",
            ).model_dump(exclude_none=True),
            "kvstore": KVStoreReference(
                backend="kv_default",
                namespace="message_batches",
            ).model_dump(exclude_none=True),
        }
