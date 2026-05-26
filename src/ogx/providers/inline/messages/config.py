# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

from pydantic import BaseModel, Field

from ogx.core.storage.datatypes import KVStoreReference


class MessagesConfig(BaseModel):
    """Configuration for the built-in Anthropic Messages API adapter."""

    kvstore: KVStoreReference = Field(
        description="Configuration for the key-value store backend used by message batches.",
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

    alias_prefixes: list[str] = Field(
        default=["claude-"],
        description=(
            "Model-ID prefixes treated as fallback aliases. Claude Code hardcodes model IDs "
            "(e.g. 'claude-haiku-4-5-...') into its background requests; when an unprefixed model "
            "with one of these prefixes is requested and is not registered, it is rewritten to the "
            "last real model the same caller used, so those requests follow the user's actual model "
            "and provider instead of failing."
        ),
    )

    fallback_model: str | None = Field(
        default=None,
        description=(
            "Model ID to use for an alias request when the caller has not yet made a request with a "
            "real model (cold start). When None, an alias request with no prior model returns an error "
            "rather than silently routing to an arbitrary, possibly expensive, provider."
        ),
    )

    @classmethod
    def sample_run_config(cls, __distro_dir__: str = "") -> dict[str, Any]:
        return {
            "kvstore": KVStoreReference(
                backend="kv_default",
                namespace="message_batches",
            ).model_dump(exclude_none=True),
        }
