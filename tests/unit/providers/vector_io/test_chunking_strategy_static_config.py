# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import pytest

from ogx_api import VectorStoreChunkingStrategyStaticConfig


def test_static_config_validation_overlap_equal_to_size():
    """chunk_overlap_tokens == max_chunk_size_tokens must be rejected.

    Regression test: unvalidated, this combination reaches
    make_overlapped_chunks()'s range(0, len(tokens), window_len - overlap_len)
    with a zero step, raising ValueError: range() arg 3 must not be zero deep
    inside file processing instead of a clear validation error at request time.
    """
    with pytest.raises(ValueError, match="chunk_overlap_tokens must be less than max_chunk_size_tokens"):
        VectorStoreChunkingStrategyStaticConfig(
            max_chunk_size_tokens=100,
            chunk_overlap_tokens=100,
        )


def test_static_config_validation_overlap_greater_than_size():
    """chunk_overlap_tokens > max_chunk_size_tokens must be rejected.

    Regression test: unvalidated, this combination gives
    make_overlapped_chunks() a negative step, so the range is empty and it
    silently returns zero chunks, discarding the document with a misleading
    "No chunks were generated from the file" error instead of a validation error.
    """
    with pytest.raises(ValueError, match="chunk_overlap_tokens must be less than max_chunk_size_tokens"):
        VectorStoreChunkingStrategyStaticConfig(
            max_chunk_size_tokens=100,
            chunk_overlap_tokens=150,
        )


def test_static_config_validation_overlap_less_than_size_is_valid():
    """A properly configured overlap/size pair must still construct normally."""
    config = VectorStoreChunkingStrategyStaticConfig(
        max_chunk_size_tokens=800,
        chunk_overlap_tokens=400,
    )
    assert config.max_chunk_size_tokens == 800
    assert config.chunk_overlap_tokens == 400


def test_static_config_default_values_are_valid():
    """The default chunk_overlap_tokens/max_chunk_size_tokens must not trip the new validator."""
    config = VectorStoreChunkingStrategyStaticConfig()
    assert config.chunk_overlap_tokens < config.max_chunk_size_tokens
