# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

r"""
Tests for metadata filter key validation (SQL injection prevention).

The `key` field in ComparisonFilter is interpolated into SQL/JSON paths in
multiple providers (sqlite_vec, milvus). A Pydantic field_validator now
rejects keys that don't match `^[a-zA-Z0-9_\-]+$`, preventing SQL injection.

Run:  uv run pytest tests/unit/providers/vector_io/test_sqlite_vec_sql_injection.py -v -s
"""

import numpy as np
import pytest
from pydantic import ValidationError

from ogx.providers.inline.vector_io.sqlite_vec.sqlite_vec import SQLiteVecIndex
from ogx.providers.utils.vector_io.vector_utils import generate_chunk_id
from ogx_api import ChunkMetadata, EmbeddedChunk
from ogx_api.filters import ComparisonFilter


def test_valid_keys_accepted():
    """Normal metadata keys should pass validation."""
    for key in ["author", "doc-id", "field_1", "Category", "x"]:
        f = ComparisonFilter(type="eq", key=key, value="test")
        assert f.key == key


@pytest.mark.parametrize(
    "malicious_key",
    [
        "'",
        "'; DROP TABLE chunks; --",
        "' OR 1=1 --",
        "key', 1) OR 1=1 OR (1=1)",
        "a;DELETE FROM t",
        "a b",
    ],
)
def test_malicious_keys_rejected(malicious_key):
    """Keys with SQL metacharacters must be rejected with a helpful message."""
    with pytest.raises(ValidationError) as exc_info:
        ComparisonFilter(type="eq", key=malicious_key, value="x")

    msg = str(exc_info.value)
    assert "Failed to validate metadata filter key" in msg
    assert repr(malicious_key) in msg


@pytest.fixture(scope="session")
def embedding_dimension():
    return 768


@pytest.fixture
async def sqlite_vec_index(embedding_dimension, tmp_path_factory):
    temp_dir = tmp_path_factory.getbasetemp()
    db_path = str(temp_dir / "test_sqli.db")
    index = await SQLiteVecIndex.create(dimension=embedding_dimension, db_path=db_path, bank_id="sqli_test")
    yield index
    await index.delete()


async def test_sqlite_vec_normal_filter_works(sqlite_vec_index):
    """End-to-end: valid filter works against the real SQLite backend."""
    chunk_id = generate_chunk_id("doc-1", "content-1")
    await sqlite_vec_index.add_chunks(
        [
            EmbeddedChunk(
                content="test",
                chunk_id=chunk_id,
                metadata={"author": "alice"},
                chunk_metadata=ChunkMetadata(
                    document_id="doc-1",
                    chunk_id=chunk_id,
                    created_timestamp=0,
                    updated_timestamp=0,
                    content_token_count=3,
                ),
                embedding=np.zeros(768, dtype=np.float32).tolist(),
                embedding_model="test",
                embedding_dimension=768,
            )
        ]
    )

    resp = await sqlite_vec_index.query_vector(
        np.zeros(768, dtype=np.float32),
        k=3,
        score_threshold=0.0,
        filters=ComparisonFilter(type="eq", key="author", value="alice"),
    )
    assert len(resp.chunks) == 1
