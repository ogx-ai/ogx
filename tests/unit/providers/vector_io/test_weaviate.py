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
from ogx.providers.utils.memory.vector_store import ChunkForDeletion
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


@pytest.fixture
def filter_by_property(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Patches the module-level Filter so the "where" Filter.by_property() builds can be
    asserted on directly, regardless of whether the real weaviate package or the stub from
    test_vector_store_kvstore_persistence.py is loaded."""
    mock = MagicMock(name="Filter")
    monkeypatch.setattr(weaviate_module, "Filter", mock)
    return mock


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


# ---------------------------------------------------------------------------
# delete() / delete_chunks() regression tests
# See: https://github.com/ogx-ai/ogx/issues/6710
#
# "id" is the Weaviate object UUID, not a stored property -- the collection only has
# chunk_id and chunk_content (see register_vector_store). delete(chunk_ids) used to filter
# on "id", which matches nothing, so it silently deleted zero objects.
# ---------------------------------------------------------------------------


async def test_delete_with_chunk_ids_filters_by_chunk_id_property(filter_by_property, index_and_collection):
    index, collection = index_and_collection

    await index.delete(chunk_ids=["chunk-a", "chunk-b"])

    filter_by_property.by_property.assert_called_once_with("chunk_id")
    filter_by_property.by_property.return_value.contains_any.assert_called_once_with(["chunk-a", "chunk-b"])
    collection.data.delete_many.assert_called_once_with(
        where=filter_by_property.by_property.return_value.contains_any.return_value
    )


async def test_delete_chunks_filters_by_chunk_id_property(filter_by_property, index_and_collection):
    index, collection = index_and_collection

    await index.delete_chunks(
        [
            ChunkForDeletion(chunk_id="chunk-a", document_id="doc-1"),
            ChunkForDeletion(chunk_id="chunk-b", document_id="doc-1"),
        ]
    )

    filter_by_property.by_property.assert_called_once_with("chunk_id")
    filter_by_property.by_property.return_value.contains_any.assert_called_once_with(["chunk-a", "chunk-b"])
    collection.data.delete_many.assert_called_once_with(
        where=filter_by_property.by_property.return_value.contains_any.return_value
    )


async def test_delete_without_chunk_ids_drops_the_collection(index_and_collection):
    index, _ = index_and_collection
    index.client.collections.exists.return_value = True

    await index.delete()

    index.client.collections.delete.assert_called_once_with(index.collection_name)
