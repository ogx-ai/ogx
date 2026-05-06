# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import time

from ogx.providers.utils.vector_io.vector_utils import (
    generate_chunk_id,
    load_embedded_chunk_with_backward_compat,
)
from ogx_api import Chunk, ChunkMetadata, VectorStoreFileObject

# This test is a unit test for the chunk_utils.py helpers. This should only contain
# tests which are specific to this file. More general (API-level) tests should be placed in
# tests/integration/vector_io/
#
# How to run this test:
#
# pytest tests/unit/providers/vector_io/test_chunk_utils.py \
# -v -s --tb=short --disable-warnings --asyncio-mode=auto


def test_generate_chunk_id():
    """Test that generate_chunk_id produces expected hashes."""
    chunk_id1 = generate_chunk_id("doc-1", "test")
    chunk_id2 = generate_chunk_id("doc-1", "test ")
    chunk_id3 = generate_chunk_id("doc-1", "test 3")

    chunk_ids = sorted([chunk_id1, chunk_id2, chunk_id3])
    assert chunk_ids == [
        "31d1f9a3-c8d2-66e7-3c37-af2acd329778",
        "d07dade7-29c0-cda7-df29-0249a1dcbc3e",
        "d14f75a1-5855-7f72-2c78-d9fc4275a346",
    ]


def test_generate_chunk_id_with_window():
    """Test that generate_chunk_id with chunk_window produces different IDs."""
    # Create a chunk object to match the original test behavior (passing object to generate_chunk_id)
    chunk = Chunk(
        content="test",
        chunk_id="placeholder",
        metadata={"document_id": "doc-1"},
        chunk_metadata=ChunkMetadata(
            document_id="doc-1",
            chunk_id="placeholder",
            created_timestamp=int(time.time()),
            updated_timestamp=int(time.time()),
            content_token_count=1,
        ),
    )
    chunk_id1 = generate_chunk_id("doc-1", chunk, chunk_window="0-1")
    chunk_id2 = generate_chunk_id("doc-1", chunk, chunk_window="1-2")
    # Verify that different windows produce different IDs
    assert chunk_id1 != chunk_id2
    assert len(chunk_id1) == 36  # Valid UUID format
    assert len(chunk_id2) == 36  # Valid UUID format


def test_chunk_creation_with_explicit_id():
    """Test that chunks can be created with explicit chunk_id."""
    chunk_id = generate_chunk_id("doc-1", "test")
    chunk = Chunk(
        content="test",
        chunk_id=chunk_id,
        metadata={"document_id": "doc-1"},
        chunk_metadata=ChunkMetadata(
            document_id="doc-1",
            chunk_id=chunk_id,
            created_timestamp=int(time.time()),
            updated_timestamp=int(time.time()),
            content_token_count=1,
        ),
    )
    assert chunk.chunk_id == chunk_id
    assert chunk.chunk_id == "31d1f9a3-c8d2-66e7-3c37-af2acd329778"


def test_chunk_with_metadata():
    """Test chunks with ChunkMetadata."""
    chunk_id = "chunk-id-1"
    chunk = Chunk(
        content="test",
        chunk_id=chunk_id,
        metadata={"document_id": "existing-id"},
        chunk_metadata=ChunkMetadata(
            document_id="document_1",
            chunk_id=chunk_id,
            created_timestamp=int(time.time()),
            updated_timestamp=int(time.time()),
            content_token_count=1,
        ),
    )
    assert chunk.chunk_id == "chunk-id-1"
    assert chunk.document_id == "existing-id"  # metadata takes precedence


def test_chunk_serialization():
    """Test that chunk_id is properly serialized."""
    chunk = Chunk(
        content="test",
        chunk_id="test-chunk-id",
        metadata={"document_id": "doc-1"},
        chunk_metadata=ChunkMetadata(
            document_id="doc-1",
            chunk_id="test-chunk-id",
            created_timestamp=int(time.time()),
            updated_timestamp=int(time.time()),
            content_token_count=1,
        ),
    )
    serialized_chunk = chunk.model_dump()
    assert serialized_chunk["chunk_id"] == "test-chunk-id"
    assert "chunk_id" in serialized_chunk


def test_vector_store_file_object_attributes_validation():
    """Test VectorStoreFileObject validates and sanitizes attributes at input boundary."""
    # Test with metadata containing lists, nested dicts, and primitives
    from ogx_api.vector_io import VectorStoreChunkingStrategyAuto

    file_obj = VectorStoreFileObject(
        id="file-123",
        attributes={
            "tags": ["transformers", "h100-compatible", "region:us"],  # List -> string
            "model_name": "granite-3.3-8b",  # String preserved
            "score": 0.95,  # Float preserved
            "active": True,  # Bool preserved
            "count": 42,  # Int -> float
            "nested": {"key": "value"},  # Dict filtered out
        },
        chunking_strategy=VectorStoreChunkingStrategyAuto(),
        created_at=1234567890,
        status="completed",
        vector_store_id="vs-123",
        usage_bytes=0,
    )

    # Lists converted to comma-separated strings
    assert file_obj.attributes["tags"] == "transformers, h100-compatible, region:us"
    # Primitives preserved
    assert file_obj.attributes["model_name"] == "granite-3.3-8b"
    assert file_obj.attributes["score"] == 0.95
    assert file_obj.attributes["active"] is True
    assert file_obj.attributes["count"] == 42.0  # int -> float
    # Complex types filtered out
    assert "nested" not in file_obj.attributes


def test_vector_store_file_object_attributes_constraints():
    """Test VectorStoreFileObject enforces OpenAPI constraints on attributes."""
    from ogx_api.vector_io import VectorStoreChunkingStrategyAuto

    # Test max 16 properties
    many_attrs = {f"key{i}": f"value{i}" for i in range(20)}
    file_obj = VectorStoreFileObject(
        id="file-123",
        attributes=many_attrs,
        chunking_strategy=VectorStoreChunkingStrategyAuto(),
        created_at=1234567890,
        status="completed",
        vector_store_id="vs-123",
        usage_bytes=0,
    )
    assert len(file_obj.attributes) == 16  # Max 16 properties

    # Test max 64 char keys are filtered
    long_key_attrs = {"a" * 65: "value", "valid_key": "value"}
    file_obj = VectorStoreFileObject(
        id="file-124",
        attributes=long_key_attrs,
        chunking_strategy=VectorStoreChunkingStrategyAuto(),
        created_at=1234567890,
        status="completed",
        vector_store_id="vs-123",
        usage_bytes=0,
    )
    assert "a" * 65 not in file_obj.attributes
    assert "valid_key" in file_obj.attributes

    # Test max 512 char string values are truncated
    long_value_attrs = {"key": "x" * 600}
    file_obj = VectorStoreFileObject(
        id="file-125",
        attributes=long_value_attrs,
        chunking_strategy=VectorStoreChunkingStrategyAuto(),
        created_at=1234567890,
        status="completed",
        vector_store_id="vs-123",
        usage_bytes=0,
    )
    assert len(file_obj.attributes["key"]) == 512


def test_load_embedded_chunk_backward_compatibility():
    """Test backward compatibility migration from legacy to current format"""
    timestamp = int(time.time())

    # Test current format (no migration needed)
    current_data = {
        "chunk_id": "current",
        "content": "test",
        "metadata": {},
        "chunk_metadata": {
            "document_id": "doc1",
            "chunk_id": "current",
            "created_timestamp": timestamp,
            "updated_timestamp": timestamp,
            "content_token_count": 1,
        },
        "embedding": [0.1, 0.2, 0.3],
        "embedding_model": "current-model",
        "embedding_dimension": 3,
    }
    chunk = load_embedded_chunk_with_backward_compat(current_data)
    assert chunk.embedding_model == "current-model"
    assert chunk.embedding_dimension == 3

    # Test legacy format (fields in chunk_metadata)
    legacy_data = {
        "chunk_id": "legacy",
        "content": "test",
        "metadata": {},
        "chunk_metadata": {
            "document_id": "doc1",
            "chunk_id": "legacy",
            "created_timestamp": timestamp,
            "updated_timestamp": timestamp,
            "content_token_count": 1,
            "chunk_embedding_model": "legacy-model",
            "chunk_embedding_dimension": 3,
        },
        "embedding": [0.4, 0.5, 0.6],
    }
    chunk = load_embedded_chunk_with_backward_compat(legacy_data)
    assert chunk.embedding_model == "legacy-model"  # Migrated
    assert chunk.embedding_dimension == 3  # Migrated


def test_load_embedded_chunk_fallbacks():
    """Test fallback behavior when embedding metadata is missing"""
    timestamp = int(time.time())

    # Test missing model (should fallback to "unknown")
    base_data = {
        "chunk_id": "fallback",
        "content": "test",
        "metadata": {},
        "chunk_metadata": {
            "document_id": "doc1",
            "chunk_id": "fallback",
            "created_timestamp": timestamp,
            "updated_timestamp": timestamp,
            "content_token_count": 1,
        },
        "embedding": [0.1, 0.2],
    }
    chunk = load_embedded_chunk_with_backward_compat(base_data)
    assert chunk.embedding_model == "unknown"
    assert chunk.embedding_dimension == 2  # Inferred from embedding length

    # Test missing embedding vector (should add empty list)
    no_embedding_data = {
        "chunk_id": "fallback",
        "content": "test",
        "metadata": {},
        "chunk_metadata": {
            "document_id": "doc1",
            "chunk_id": "fallback",
            "created_timestamp": timestamp,
            "updated_timestamp": timestamp,
            "content_token_count": 1,
        },
    }
    chunk = load_embedded_chunk_with_backward_compat(no_embedding_data)
    assert chunk.embedding_model == "unknown"
    assert chunk.embedding_dimension == 0
    assert chunk.embedding == []
