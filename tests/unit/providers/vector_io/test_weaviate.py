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
from ogx.providers.remote.vector_io.weaviate.config import WeaviateVectorIOConfig
from ogx.providers.remote.vector_io.weaviate.weaviate import WeaviateIndex, WeaviateVectorIOAdapter
from ogx.providers.utils.memory.vector_store import ChunkForDeletion
from ogx_api import ChunkMetadata, EmbeddedChunk, VectorStore

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
# ---------------------------------------------------------------------------


def _assert_filter_on_chunk_ids(filter_arg, chunk_ids: list[str]) -> None:
    expected_uuids = [_expected_uuid(chunk_id) for chunk_id in chunk_ids]
    if isinstance(filter_arg, MagicMock):
        weaviate_module.Filter.by_id.return_value.contains_any.assert_called_with(expected_uuids)
    else:
        assert filter_arg.target == "_id"
        assert filter_arg.value == expected_uuids


async def test_delete_chunks_filters_on_chunk_id(index_and_collection):
    index, collection = index_and_collection

    await index.delete_chunks(
        [
            ChunkForDeletion(chunk_id="chunk-a", document_id="doc-1"),
            ChunkForDeletion(chunk_id="chunk-b", document_id="doc-1"),
        ]
    )

    collection.data.delete_many.assert_called_once()
    filter_arg = collection.data.delete_many.call_args.kwargs["where"]
    _assert_filter_on_chunk_ids(filter_arg, ["chunk-a", "chunk-b"])


async def test_delete_with_chunk_ids_filters_on_chunk_id(index_and_collection):
    index, collection = index_and_collection

    await index.delete(chunk_ids=["chunk-a", "chunk-b"])

    collection.data.delete_many.assert_called_once()
    filter_arg = collection.data.delete_many.call_args.kwargs["where"]
    _assert_filter_on_chunk_ids(filter_arg, ["chunk-a", "chunk-b"])


async def test_delete_without_chunk_ids_drops_the_collection(index_and_collection):
    index, _ = index_and_collection
    index.client.collections.exists.return_value = True

    await index.delete()

    index.client.collections.delete.assert_called_once_with(index.collection_name)


async def test_delete_with_empty_chunk_ids_is_a_noop(index_and_collection):
    """Filter.contains_any([]) raises WeaviateInvalidInputError (verified against a real
    Weaviate 1.27.1 server) rather than matching nothing, so an empty list must short-circuit
    before reaching the client."""
    index, collection = index_and_collection

    await index.delete(chunk_ids=[])

    collection.data.delete_many.assert_not_called()


async def test_delete_chunks_with_empty_list_is_a_noop(index_and_collection):
    index, collection = index_and_collection

    await index.delete_chunks([])

    collection.data.delete_many.assert_not_called()


# ---------------------------------------------------------------------------
# register_vector_store chunk_id tokenization regression test
# See: community review of #6710 on PR #6715.
#
# chunk_id was left undeclared, so Weaviate's auto-schema gives it the default "word"
# tokenization, which splits on non-alphanumeric characters: "doc-1_1" and "doc-1_2" both
# tokenize to include "doc" and "1". Verified against a real Weaviate 1.27.1 server that this
# makes delete()/delete_chunks()'s contains_any filter over-match -- deleting "doc-1_1" also
# deleted "doc-1_2" -- and that declaring chunk_id with FIELD (exact) tokenization fixes it.
# ---------------------------------------------------------------------------


@pytest.fixture
def property_mock(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Patches the module-level Property constructor so its call kwargs can be asserted on
    directly (the same reasoning as the data_object fixture above: whether the real weaviate
    package or the test_vector_store_kvstore_persistence.py stub is loaded, Property() calls
    land on this mock either way, so inspecting call_args_list is robust to either)."""
    mock = MagicMock(name="Property")
    monkeypatch.setattr(weaviate_module.wvc.config, "Property", mock)
    return mock


@pytest.fixture
def weaviate_adapter(monkeypatch: pytest.MonkeyPatch) -> tuple[WeaviateVectorIOAdapter, MagicMock]:
    client = MagicMock()
    client.collections.exists.return_value = False
    adapter = WeaviateVectorIOAdapter(config=WeaviateVectorIOConfig(), inference_api=MagicMock(), files_api=None)
    monkeypatch.setattr(adapter, "_get_client", lambda: client)
    return adapter, client


async def test_register_vector_store_declares_chunk_id_with_field_tokenization(property_mock, weaviate_adapter):
    adapter, _ = weaviate_adapter

    await adapter.register_vector_store(
        VectorStore(
            identifier="test-store",
            provider_id="weaviate",
            embedding_model="test-model",
            embedding_dimension=3,
        )
    )

    chunk_id_call = next(call for call in property_mock.call_args_list if call.kwargs.get("name") == "chunk_id")
    assert chunk_id_call.kwargs["tokenization"] == weaviate_module.wvc.config.Tokenization.FIELD
