# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

from pydantic import BaseModel, Field, SecretStr

from ogx_api.vector_io import VectorStoreChunkingStrategyStaticConfig


class UnstructuredApiFileProcessorConfig(BaseModel):
    """Configuration for Unstructured.io API file processor."""

    api_key: SecretStr = Field(
        description="API key for authenticating with Unstructured.io SaaS API (get one from https://unstructured.io)"
    )
    default_chunk_size_tokens: int = Field(
        default=VectorStoreChunkingStrategyStaticConfig.model_fields["max_chunk_size_tokens"].default,
        ge=100,
        le=4096,
        description="Default chunk size in tokens when chunking_strategy type is 'auto'",
    )

    @classmethod
    def sample_run_config(cls, **kwargs: Any) -> dict[str, Any]:
        return {
            "api_key": "${env.UNSTRUCTURED_API_KEY}",
        }
