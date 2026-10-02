# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import hashlib
import uuid
from unittest.mock import MagicMock

import pytest

from ogx.providers.remote.vector_io.weaviate import weaviate as weaviate_module
from ogx.providers.remote.vector_io.weaviate.weaviate import WeaviateIndex
from ogx_api import ChunkMetadata, EmbeddedChunk

# These tests stay database-free: they check the objects handed to the Weaviate
# client. Weaviate batch imports overwrite an existing object that has the same
# UUID, so a deterministic UUID per chunk_id gives upsert semantics (#6256).
# DataObject is patched so the tests behave the same whether the real weaviate
# package or the stub from test_vector_store_kvstore_persistence.py is loaded.


def _expected_uuid(chunk_id: str) -> str:
    sha256_hash = hashlib.sha256(chunk_id.encode()).hexdigest()
    return str(uuid.UUID(sha256_hash[:32]))


def _chunk(chunk_id: str, content: str) -> EmbeddedChunk:
    return EmbeddedChunk(
        content=content,
        chunk_id=chunk_id,
        metadata={"document_id": "doc-1"},
        chunk_metadata=ChunkMetadata(document_id="doc-1", chunk_id=chunk_id),
        embedding=[0.1, 0.2, 0.3],
        embedding_model="test-model",
        embedding_dimension=3,
    )


@pytest.fixture
def data_object(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    mock = MagicMock(name="DataObject")
    monkeypatch.setattr(weaviate_module.wvc.data, "DataObject", mock)
    return mock


@pytest.fixture
def index_and_collection() -> tuple[WeaviateIndex, MagicMock]:
    client = MagicMock()
    collection = client.collections.get.return_value
    return WeaviateIndex(client=client, collection_name="test_collection"), collection


async def test_add_chunks_uses_uuid_derived_from_chunk_id(data_object, index_and_collection):
    index, collection = index_and_collection

    await index.add_chunks([_chunk("chunk-a", "first"), _chunk("chunk-b", "second")])

    uuids = [call.kwargs["uuid"] for call in data_object.call_args_list]
    assert uuids == [_expected_uuid("chunk-a"), _expected_uuid("chunk-b")]
    collection.data.insert_many.assert_called_once()


async def test_reinserting_same_chunk_id_targets_same_object(data_object, index_and_collection):
    index, _ = index_and_collection

    await index.add_chunks([_chunk("chunk-a", "old text")])
    await index.add_chunks([_chunk("chunk-a", "new text")])

    first, second = data_object.call_args_list
    assert first.kwargs["uuid"] == second.kwargs["uuid"] == _expected_uuid("chunk-a")
    assert '"new text"' in second.kwargs["properties"]["chunk_content"]


async def test_add_chunks_empty_list_is_noop(data_object, index_and_collection):
    index, collection = index_and_collection

    await index.add_chunks([])

    data_object.assert_not_called()
    collection.data.insert_many.assert_not_called()
