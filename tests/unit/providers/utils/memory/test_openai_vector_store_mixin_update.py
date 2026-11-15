# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

# Reproducer for https://github.com/ogx-ai/ogx/issues/7002.
#
# ``openai_attach_file_to_vector_store`` bakes the file's attributes into every
# chunk's metadata, and both search filters and the attributes of search
# results read that chunk metadata. ``openai_update_vector_store_file`` used to
# update only the file record, so after an update the index kept filtering on
# (and returning) the attach-time values: a filter on the new value matched
# nothing and a filter on the old value still matched.
#
# The tests below pin the desired behavior: the update propagates into the
# stored chunks and into the vector index (delete-then-insert with the stored
# embeddings, so no re-embedding), old attribute keys are removed, and
# attach-level metadata (filename, file_id, processor keys) survives.

import asyncio
from unittest.mock import AsyncMock

from ogx.providers.utils.memory.openai_vector_store_mixin import OpenAIVectorStoreMixin
from ogx_api import OpenAIUpdateVectorStoreFileRequest, VectorStoreChunkingStrategyAuto
from ogx_api.vector_io.models import ChunkMetadata, EmbeddedChunk


def _make_store_info():
    """Build a minimal in-memory vector store dict matching the mixin's expectations."""
    return {
        "file_ids": [],
        "file_counts": {"total": 0, "completed": 0, "cancelled": 0, "failed": 0, "in_progress": 0},
        "metadata": {},
    }


class MockVectorStoreMixin(OpenAIVectorStoreMixin):
    """Mock implementation of OpenAIVectorStoreMixin for testing."""

    def __init__(self, inference_api, files_api, kvstore=None, file_processor_api=None, metadata_store=None):
        super().__init__(
            inference_api=inference_api,
            files_api=files_api,
            kvstore=kvstore,
            file_processor_api=file_processor_api,
            metadata_store=metadata_store,
        )

    async def register_vector_store(self, vector_store):
        pass

    async def unregister_vector_store(self, vector_store_id):
        pass

    async def insert_chunks(self, request):
        pass

    async def query_chunks(self, request):
        pass

    async def delete_chunks(self, request):
        pass


class _DictKVStore:
    """Minimal in-memory kvstore good enough for the mixin's file storage paths."""

    def __init__(self):
        self.store: dict[str, str] = {}

    async def set(self, key: str, value: str) -> None:
        self.store[key] = value

    async def get(self, key: str) -> str | None:
        return self.store.get(key)

    async def delete(self, key: str) -> None:
        self.store.pop(key, None)

    async def values_in_range(self, start_key: str, end_key: str) -> list[str]:
        return [v for k, v in sorted(self.store.items()) if start_key <= k < end_key]


class TestUpdateVectorStoreFileAttributePropagation:
    """Updating a file's attributes must reach the chunks search filters read."""

    def _make_mixin(self):
        mixin = MockVectorStoreMixin(
            inference_api=AsyncMock(),
            files_api=AsyncMock(),
            kvstore=_DictKVStore(),
        )
        mixin.deleted_requests = []
        mixin.inserted_requests = []

        async def record_delete(request):
            mixin.deleted_requests.append(request)

        async def record_insert(request):
            mixin.inserted_requests.append(request)

        mixin.delete_chunks = record_delete
        mixin.insert_chunks = record_insert
        return mixin

    def _make_chunk(self, file_id: str, idx: int, attributes: dict) -> EmbeddedChunk:
        metadata = {"page": idx, "document_id": file_id, **attributes}
        metadata["filename"] = "real.pdf"
        metadata["file_id"] = file_id
        return EmbeddedChunk(
            content=f"chunk text {idx}",
            chunk_id=f"{file_id}-chunk-{idx}",
            metadata=metadata,
            chunk_metadata=ChunkMetadata(document_id=file_id),
            embedding=[float(idx), 0.5],
            embedding_model="test-model",
            embedding_dimension=2,
        )

    async def _seed_file(self, mixin, attributes: dict, n_chunks: int = 2):
        vector_store_id = "vs_1"
        file_id = "file_1"
        store_info = _make_store_info()
        store_info["file_ids"] = [file_id]
        mixin.openai_vector_stores[vector_store_id] = store_info
        file_info = {
            "id": file_id,
            "status": "completed",
            "vector_store_id": vector_store_id,
            "attributes": attributes,
            "filename": "real.pdf",
            "created_at": 1,
            "usage_bytes": 10,
            "chunking_strategy": VectorStoreChunkingStrategyAuto().model_dump(),
        }
        chunks = [self._make_chunk(file_id, i, attributes) for i in range(n_chunks)]
        await mixin._save_openai_vector_store_file(
            vector_store_id, file_id, file_info, [c.model_dump() for c in chunks]
        )
        return vector_store_id, file_id, chunks

    async def test_update_reaches_stored_chunks_and_index(self):
        mixin = self._make_mixin()
        vector_store_id, file_id, chunks = await self._seed_file(mixin, {"kind": "a", "old": 1})

        result = await mixin.openai_update_vector_store_file(
            vector_store_id=vector_store_id,
            file_id=file_id,
            request=OpenAIUpdateVectorStoreFileRequest(attributes={"kind": "b"}),
        )

        assert result.attributes == {"kind": "b"}

        # The file record and its stored chunks carry the new attributes, old keys
        # are gone, and attach-level metadata is untouched.
        stored = await mixin._load_openai_vector_store_file_contents(vector_store_id, file_id)
        assert len(stored) == 2
        for c in stored:
            assert c["metadata"]["kind"] == "b"
            assert "old" not in c["metadata"]
            assert c["metadata"]["filename"] == "real.pdf"
            assert c["metadata"]["file_id"] == file_id
        assert stored[0]["metadata"]["page"] == 0
        assert stored[1]["metadata"]["page"] == 1

        # The index saw delete-then-insert with the same ids and embeddings.
        assert len(mixin.deleted_requests) == 1
        assert {c.chunk_id for c in mixin.deleted_requests[0].chunks} == {c.chunk_id for c in chunks}
        assert all(c.document_id == file_id for c in mixin.deleted_requests[0].chunks)
        assert len(mixin.inserted_requests) == 1
        for new_chunk, old_chunk in zip(mixin.inserted_requests[0].chunks, chunks, strict=True):
            assert new_chunk.chunk_id == old_chunk.chunk_id
            assert new_chunk.embedding == old_chunk.embedding
            assert new_chunk.metadata["kind"] == "b"

        # Attribute propagation reuses stored embeddings; no re-embedding happens.
        mixin.inference_api.openai_embeddings.assert_not_called()

    async def test_update_with_empty_attributes_clears_old_keys(self):
        mixin = self._make_mixin()
        vector_store_id, file_id, _ = await self._seed_file(mixin, {"kind": "a", "old": 1})

        await mixin.openai_update_vector_store_file(
            vector_store_id=vector_store_id,
            file_id=file_id,
            request=OpenAIUpdateVectorStoreFileRequest(attributes={}),
        )

        stored = await mixin._load_openai_vector_store_file_contents(vector_store_id, file_id)
        for c in stored:
            assert "kind" not in c["metadata"]
            assert "old" not in c["metadata"]
            assert c["metadata"]["filename"] == "real.pdf"
        assert len(mixin.inserted_requests) == 1

    async def test_update_does_not_clobber_attach_level_filename_and_file_id(self):
        mixin = self._make_mixin()
        vector_store_id, file_id, _ = await self._seed_file(
            mixin, {"filename": "user-supplied.pdf", "file_id": "user-supplied-id", "kind": "a"}
        )

        await mixin.openai_update_vector_store_file(
            vector_store_id=vector_store_id,
            file_id=file_id,
            request=OpenAIUpdateVectorStoreFileRequest(attributes={"kind": "b"}),
        )

        stored = await mixin._load_openai_vector_store_file_contents(vector_store_id, file_id)
        for c in stored:
            assert c["metadata"]["filename"] == "real.pdf"
            assert c["metadata"]["file_id"] == file_id
            assert c["metadata"]["kind"] == "b"

    async def test_update_skips_index_when_file_not_completed(self):
        mixin = self._make_mixin()
        vector_store_id, file_id, _ = await self._seed_file(mixin, {"kind": "a"})
        file_info = await mixin._load_openai_vector_store_file(vector_store_id, file_id)
        file_info["status"] = "failed"
        await mixin._update_openai_vector_store_file(vector_store_id, file_id, file_info)

        result = await mixin.openai_update_vector_store_file(
            vector_store_id=vector_store_id,
            file_id=file_id,
            request=OpenAIUpdateVectorStoreFileRequest(attributes={"kind": "b"}),
        )

        assert result.attributes == {"kind": "b"}
        assert mixin.deleted_requests == []
        assert mixin.inserted_requests == []

    async def test_concurrent_updates_leave_coherent_chunk_metadata(self):
        mixin = self._make_mixin()
        vector_store_id, file_id, _ = await self._seed_file(mixin, {})

        # Force both updates to yield inside their critical sections; without the
        # store lock the two writers can interleave delete/insert and leak keys.
        original_insert = mixin.insert_chunks

        async def slow_insert(request):
            await asyncio.sleep(0)
            await original_insert(request)

        mixin.insert_chunks = slow_insert

        await asyncio.gather(
            mixin.openai_update_vector_store_file(
                vector_store_id=vector_store_id,
                file_id=file_id,
                request=OpenAIUpdateVectorStoreFileRequest(attributes={"x": 1}),
            ),
            mixin.openai_update_vector_store_file(
                vector_store_id=vector_store_id,
                file_id=file_id,
                request=OpenAIUpdateVectorStoreFileRequest(attributes={"y": 2}),
            ),
        )

        stored = await mixin._load_openai_vector_store_file_contents(vector_store_id, file_id)
        file_info = await mixin._load_openai_vector_store_file(vector_store_id, file_id)
        # The second serialized writer drops the first writer's keys, so the final
        # state is exactly one coherent attribute set on both the record and chunks.
        assert file_info["attributes"] in ({"x": 1}, {"y": 2})
        for c in stored:
            assert not ({"x", "y"} <= set(c["metadata"]))
            assert c["metadata"].get("x") == file_info["attributes"].get("x")
            assert c["metadata"].get("y") == file_info["attributes"].get("y")
