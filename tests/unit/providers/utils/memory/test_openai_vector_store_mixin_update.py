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
import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import numpy as np
import pytest

from ogx.core.storage.datatypes import SqliteSqlStoreConfig
from ogx.core.storage.sqlstore.authorized_sqlstore import AuthorizedSqlStore
from ogx.core.storage.sqlstore.sqlalchemy_sqlstore import SqlAlchemySqlStoreImpl
from ogx.providers.inline.vector_io.faiss.faiss import FaissIndex
from ogx.providers.utils.memory.openai_vector_store_mixin import OpenAIVectorStoreMixin
from ogx.providers.utils.vector_io.filters import parse_filter
from ogx_api import (
    DeleteChunksRequest,
    InsertChunksRequest,
    OpenAIUpdateVectorStoreFileRequest,
    VectorStoreChunkingStrategyAuto,
)
from ogx_api.vector_io.models import ChunkMetadata, EmbeddedChunk


@pytest.fixture(params=["kvstore", "sqlstore"])
async def metadata_store(request: pytest.FixtureRequest, tmp_path: Path) -> AsyncIterator[AuthorizedSqlStore | None]:
    if request.param == "kvstore":
        yield None
    else:
        sql = SqlAlchemySqlStoreImpl(SqliteSqlStoreConfig(db_path=str(tmp_path / "metadata.db")))
        try:
            yield AuthorizedSqlStore(sql_store=sql, policy=[])
        finally:
            await sql.shutdown()


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

    async def keys_in_range(self, start_key: str, end_key: str) -> list[str]:
        return [k for k in sorted(self.store) if start_key <= k < end_key]


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
        store_info["file_counts"].update(total=1, completed=1)
        store_info["usage_bytes"] = 10
        await mixin._save_openai_vector_store(vector_store_id, store_info)
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

    async def _use_real_faiss(self, mixin: MockVectorStoreMixin, chunks: list[EmbeddedChunk]) -> FaissIndex:
        index = await FaissIndex.create(dimension=2, kvstore=mixin.kvstore, bank_id="vs_1")
        await index.add_chunks(chunks)

        async def insert(request: InsertChunksRequest) -> None:
            await index.add_chunks(request.chunks)

        async def delete(request: DeleteChunksRequest) -> None:
            await index.delete_chunks(request.chunks)

        mixin.insert_chunks = insert
        mixin.delete_chunks = delete
        return index

    @pytest.mark.parametrize("other_file", [False, True])
    async def test_update_real_faiss_keeps_vectors_and_metadata_aligned(self, other_file: bool) -> None:
        mixin = self._make_mixin()
        vector_store_id, file_id, chunks = await self._seed_file(mixin, {"kind": "a"}, n_chunks=1)
        other = self._make_chunk("file_2", 10, {"kind": "other"})
        # Put the updated file last to exercise removal of the final position.
        all_chunks = [other, *chunks] if other_file else chunks
        index = await self._use_real_faiss(mixin, all_chunks)

        for kind in ("b", "c"):
            await mixin.openai_update_vector_store_file(
                vector_store_id, file_id, OpenAIUpdateVectorStoreFileRequest(attributes={"kind": kind})
            )
            assert index.index.ntotal == len(all_chunks)
            assert sorted(index.chunk_by_index) == list(range(len(all_chunks)))
            assert index.chunk_ids == [index.chunk_by_index[i].chunk_id for i in range(len(all_chunks))]
            for i, chunk in index.chunk_by_index.items():
                np.testing.assert_allclose(index.index.reconstruct(i), chunk.embedding)
            result = await index.query_vector(np.array(chunks[0].embedding), k=10, score_threshold=0.0)
            assert result.chunks[0].metadata["kind"] == kind
            result = await index.query_vector(
                np.array(chunks[0].embedding),
                k=10,
                score_threshold=0.0,
                filters=parse_filter({"type": "eq", "key": "kind", "value": kind}),
            )
            assert [c.chunk_id for c in result.chunks] == [chunks[0].chunk_id]
            stale = await index.query_vector(
                np.array(chunks[0].embedding),
                k=10,
                score_threshold=0.0,
                filters=parse_filter({"type": "eq", "key": "kind", "value": "a"}),
            )
            assert stale.chunks == []
            if other_file:
                assert next(c for c in index.chunk_by_index.values() if c.chunk_id == other.chunk_id) == other
            # Exercise persisted vectors and rebuilt metadata lookup too.
            index = await self._use_real_faiss(mixin, [])

    @pytest.mark.parametrize("n_chunks", [2, 12])
    @pytest.mark.parametrize("failure_stage", ["delete", "insert", "contents", "commit"])
    @pytest.mark.parametrize("retry_attributes", [{"kind": "b", "temporary": 1}, {"final": True}])
    async def test_failed_update_retry_removes_all_stale_attributes(
        self,
        failure_stage: str,
        retry_attributes: dict[str, Any],
        metadata_store: AuthorizedSqlStore | None,
        n_chunks: int,
    ) -> None:
        mixin = self._make_mixin()
        mixin.metadata_store = metadata_store
        if metadata_store:
            await mixin._create_metadata_tables()
        vector_store_id, file_id, chunks = await self._seed_file(
            mixin, {"kind": "a", "obsolete": True}, n_chunks=n_chunks
        )
        failure_index = min(3, n_chunks - 1)
        await self._use_real_faiss(mixin, chunks)
        original_delete = mixin.delete_chunks
        original_insert = mixin.insert_chunks
        original_set = mixin.kvstore.set
        armed = True

        async def fail_delete(request: DeleteChunksRequest) -> None:
            nonlocal armed
            if failure_stage == "delete" and armed:
                armed = False
                raise RuntimeError("injected failure")
            await original_delete(request)

        async def fail_insert(request: InsertChunksRequest) -> None:
            nonlocal armed
            if failure_stage == "insert" and armed:
                armed = False
                raise RuntimeError("injected failure")
            await original_insert(request)

        async def fail_set(key: str, value: str) -> None:
            nonlocal armed
            if armed and (
                (failure_stage == "contents" and "files_contents:" in key and key.endswith(f":{failure_index}"))
                or (
                    failure_stage == "commit" and "files:" in key and "_pending_attribute_keys" not in json.loads(value)
                )
            ):
                armed = False
                raise RuntimeError("injected failure")
            await original_set(key, value)

        if metadata_store:
            original_upsert = metadata_store.upsert
            original_update = metadata_store.update

            async def fail_upsert(**kwargs: Any) -> None:
                nonlocal armed
                if failure_stage == "contents" and armed and kwargs["data"].get("chunk_index") == failure_index:
                    armed = False
                    raise RuntimeError("injected failure")
                await original_upsert(**kwargs)

            async def fail_update(**kwargs: Any) -> None:
                nonlocal armed
                if failure_stage == "commit" and armed and "_pending_attribute_keys" not in kwargs["data"]["file_data"]:
                    armed = False
                    raise RuntimeError("injected failure")
                await original_update(**kwargs)

            metadata_store.upsert = fail_upsert
            metadata_store.update = fail_update
        else:
            mixin.kvstore.set = fail_set
        mixin.delete_chunks = fail_delete
        mixin.insert_chunks = fail_insert
        with pytest.raises(RuntimeError, match="injected failure"):
            await mixin.openai_update_vector_store_file(
                vector_store_id,
                file_id,
                OpenAIUpdateVectorStoreFileRequest(attributes={"kind": "b", "temporary": 1}),
            )
        failed_record = await mixin._load_openai_vector_store_file(vector_store_id, file_id)
        assert failed_record["attributes"] == {"kind": "a", "obsolete": True}
        # A fresh instance must be able to recover using only durable state.
        restarted = self._make_mixin()
        restarted.kvstore = mixin.kvstore
        restarted.metadata_store = metadata_store
        restarted.openai_vector_stores = mixin.openai_vector_stores
        index = await self._use_real_faiss(restarted, [])
        await restarted.openai_update_vector_store_file(
            vector_store_id, file_id, OpenAIUpdateVectorStoreFileRequest(attributes=retry_attributes)
        )
        stored = await restarted._load_openai_vector_store_file_contents(vector_store_id, file_id)
        assert [c["chunk_id"] for c in stored] == [c.chunk_id for c in chunks]
        for c in stored:
            assert {
                k: v for k, v in c["metadata"].items() if k in {"kind", "obsolete", "temporary", "final"}
            } == retry_attributes
        record = await restarted._load_openai_vector_store_file(vector_store_id, file_id)
        assert record["attributes"] == retry_attributes
        assert "_pending_attribute_keys" not in record
        assert index.index.ntotal == len(chunks)
        results = await index.query_vector(np.array(chunks[0].embedding), k=n_chunks, score_threshold=0.0)
        assert {c.chunk_id for c in results.chunks} == {c.chunk_id for c in chunks}
        for chunk in results.chunks:
            assert {
                k: v for k, v in chunk.metadata.items() if k in {"kind", "obsolete", "temporary", "final"}
            } == retry_attributes

    async def test_delete_waits_for_update_and_removes_everything(self) -> None:
        mixin = self._make_mixin()
        vector_store_id, file_id, chunks = await self._seed_file(mixin, {"kind": "a"})
        index = await self._use_real_faiss(mixin, chunks)
        inserting = asyncio.Event()
        resume = asyncio.Event()
        deleting = asyncio.Event()
        original_insert = mixin.insert_chunks
        original_lock = mixin._get_vector_store_lock

        async def paused_insert(request: InsertChunksRequest) -> None:
            inserting.set()
            await resume.wait()
            await original_insert(request)

        def acquire_lock(store_id: str) -> asyncio.Lock:
            deleting.set()
            return original_lock(store_id)

        mixin.insert_chunks = paused_insert
        update_task = asyncio.create_task(
            mixin.openai_update_vector_store_file(
                vector_store_id, file_id, OpenAIUpdateVectorStoreFileRequest(attributes={"kind": "b"})
            )
        )
        await asyncio.wait_for(inserting.wait(), timeout=5)
        mixin._get_vector_store_lock = acquire_lock
        delete_task = asyncio.create_task(mixin.openai_delete_vector_store_file(vector_store_id, file_id))
        await asyncio.wait_for(deleting.wait(), timeout=5)
        await asyncio.sleep(0)
        resume.set()
        await asyncio.wait_for(asyncio.gather(update_task, delete_task), timeout=5)
        assert await mixin._load_openai_vector_store_file(vector_store_id, file_id) == {}
        assert await mixin._load_openai_vector_store_file_contents(vector_store_id, file_id) == []
        assert index.index.ntotal == 0
        assert index.chunk_by_index == {}
        store_info = mixin.openai_vector_stores[vector_store_id]
        assert file_id not in store_info["file_ids"]
        assert store_info["file_counts"]["completed"] == store_info["file_counts"]["total"] == 0
        assert store_info["usage_bytes"] == 0

    async def test_update_rechecks_membership_after_waiting_for_delete(self) -> None:
        mixin = self._make_mixin()
        vector_store_id, file_id, _ = await self._seed_file(mixin, {"kind": "a"})
        authorized = asyncio.Event()
        original_lock = mixin._get_vector_store_lock

        def acquire_lock(store_id: str) -> asyncio.Lock:
            authorized.set()
            return original_lock(store_id)

        async with mixin._get_vector_store_lock(vector_store_id):
            mixin._get_vector_store_lock = acquire_lock
            task = asyncio.create_task(
                mixin.openai_update_vector_store_file(
                    vector_store_id, file_id, OpenAIUpdateVectorStoreFileRequest(attributes={"kind": "b"})
                )
            )
            await asyncio.wait_for(authorized.wait(), timeout=5)
            # Model a deletion finishing while the updater waits for its lock.
            mixin.openai_vector_stores[vector_store_id]["file_ids"].remove(file_id)
            await mixin._delete_openai_vector_store_file_from_storage(vector_store_id, file_id)
        with pytest.raises(ValueError, match="not found in vector store"):
            await task
        assert await mixin._load_openai_vector_store_file(vector_store_id, file_id) == {}
