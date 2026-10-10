# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import pytest
from pydantic import ValidationError

from ogx_api.vector_io.models import (
    DEFAULT_CHUNK_OVERLAP_TOKENS,
    DEFAULT_CHUNK_SIZE_TOKENS,
    VectorStoreChunkingStrategyStaticConfig,
)


def test_static_chunking_config_defaults():
    """Verify default static chunking configuration values."""
    config = VectorStoreChunkingStrategyStaticConfig()
    assert config.chunk_overlap_tokens == DEFAULT_CHUNK_OVERLAP_TOKENS
    assert config.max_chunk_size_tokens == DEFAULT_CHUNK_SIZE_TOKENS


def test_static_chunking_config_valid_custom():
    """Verify custom valid static chunking configuration."""
    config = VectorStoreChunkingStrategyStaticConfig(
        max_chunk_size_tokens=500,
        chunk_overlap_tokens=200,
    )
    assert config.max_chunk_size_tokens == 500
    assert config.chunk_overlap_tokens == 200


def test_static_chunking_config_rejects_equal_overlap_and_size():
    """Verify that chunk_overlap_tokens == max_chunk_size_tokens raises ValueError."""
    with pytest.raises(ValueError, match="chunk_overlap_tokens must be less than max_chunk_size_tokens"):
        VectorStoreChunkingStrategyStaticConfig(
            max_chunk_size_tokens=500,
            chunk_overlap_tokens=500,
        )


def test_static_chunking_config_rejects_greater_overlap():
    """Verify that chunk_overlap_tokens > max_chunk_size_tokens raises ValueError."""
    with pytest.raises(ValueError, match="chunk_overlap_tokens must be less than max_chunk_size_tokens"):
        VectorStoreChunkingStrategyStaticConfig(
            max_chunk_size_tokens=300,
            chunk_overlap_tokens=400,
        )


def test_static_chunking_config_bounds_max_chunk_size():
    """Verify bounds constraints on max_chunk_size_tokens (100 to 4096)."""
    with pytest.raises(ValidationError):
        VectorStoreChunkingStrategyStaticConfig(max_chunk_size_tokens=50)

    with pytest.raises(ValidationError):
        VectorStoreChunkingStrategyStaticConfig(max_chunk_size_tokens=5000)
