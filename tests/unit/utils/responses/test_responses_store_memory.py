# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from tempfile import TemporaryDirectory
from uuid import uuid4

import pytest

from ogx.core.storage.datatypes import ResponsesStoreReference, SqliteSqlStoreConfig
from ogx.core.storage.sqlstore.sqlstore import register_sqlstore_backends
from ogx.providers.utils.responses.responses_store import ResponsesStore


def build_store(db_path: str) -> ResponsesStore:
    backend_name = f"sql_responses_memory_{uuid4().hex}"
    return build_store_with_backend(db_path, backend_name)


def build_store_with_backend(db_path: str, backend_name: str) -> ResponsesStore:
    register_sqlstore_backends({backend_name: SqliteSqlStoreConfig(db_path=db_path)})
    return ResponsesStore(
        ResponsesStoreReference(backend=backend_name, table_name="responses"),
        policy=[],
    )


async def test_memory_record_upsert_replaces_current_file():
    with TemporaryDirectory() as tmpdir:
        store = build_store(f"{tmpdir}/responses.db")
        await store.initialize()

        await store.upsert_memory_record(
            owner_id="user-123",
            conversation_id="conv_abc",
            vector_store_id="vs_mem",
            file_id="file_old",
            response_id="resp_old",
        )
        first = await store.get_memory_record(
            owner_id="user-123",
            conversation_id="conv_abc",
            vector_store_id="vs_mem",
        )

        await store.upsert_memory_record(
            owner_id="user-123",
            conversation_id="conv_abc",
            vector_store_id="vs_mem",
            file_id="file_new",
            response_id="resp_new",
        )
        second = await store.get_memory_record(
            owner_id="user-123",
            conversation_id="conv_abc",
            vector_store_id="vs_mem",
        )

        assert first is not None
        assert second is not None
        assert first.file_id == "file_old"
        assert second.file_id == "file_new"
        assert second.response_id == "resp_new"
        assert second.created_at == first.created_at
        assert second.updated_at >= first.updated_at


async def test_memory_upserts_do_not_fetch_existing_rows(monkeypatch):
    with TemporaryDirectory() as tmpdir:
        store = build_store(f"{tmpdir}/responses.db")
        await store.initialize()

        async def fail_fetch_one(*args, **kwargs):
            raise AssertionError("upsert should not prefetch existing rows")

        monkeypatch.setattr(store.sql_store, "fetch_one", fail_fetch_one)

        await store.upsert_memory_record(
            owner_id="user-123",
            conversation_id="conv_abc",
            vector_store_id="vs_mem",
            file_id="file_new",
            response_id="resp_new",
        )
        await store.upsert_default_memory_vector_store(
            namespace="default",
            vector_store_id="vs_default",
            provider_id="vector-provider",
        )


async def test_default_memory_vector_store_mapping_is_shared_per_namespace_across_store_restarts():
    with TemporaryDirectory() as tmpdir:
        db_path = f"{tmpdir}/responses.db"
        store = build_store(db_path)
        await store.initialize()

        await store.upsert_default_memory_vector_store(
            namespace="default",
            vector_store_id="vs_default",
            provider_id="vector-provider",
        )

        restarted_store = build_store(db_path)
        await restarted_store.initialize()
        record = await restarted_store.get_default_memory_vector_store(
            namespace="default",
        )

        assert record is not None
        assert record.namespace == "default"
        assert record.vector_store_id == "vs_default"
        assert record.provider_id == "vector-provider"


async def test_default_memory_vector_store_claim_allows_only_one_creator():
    with TemporaryDirectory() as tmpdir:
        db_path = f"{tmpdir}/responses.db"
        backend_name = f"sql_responses_memory_{uuid4().hex}"
        first_store = build_store_with_backend(db_path, backend_name)
        second_store = build_store_with_backend(db_path, backend_name)
        await first_store.initialize()
        await second_store.initialize()

        first_claim = await first_store.claim_default_memory_vector_store(namespace="default")
        second_claim = await second_store.claim_default_memory_vector_store(namespace="default")

        assert first_claim is True
        assert second_claim is False
        pending_record = await second_store.get_default_memory_vector_store(namespace="default")
        assert pending_record is not None
        assert pending_record.vector_store_id is None

        await first_store.upsert_default_memory_vector_store(
            namespace="default",
            vector_store_id="vs_default",
            provider_id="vector-provider",
        )
        ready_record = await second_store.get_default_memory_vector_store(namespace="default")

        assert ready_record is not None
        assert ready_record.vector_store_id == "vs_default"


async def test_default_memory_vector_store_claim_can_be_released_after_failed_creation():
    with TemporaryDirectory() as tmpdir:
        db_path = f"{tmpdir}/responses.db"
        backend_name = f"sql_responses_memory_{uuid4().hex}"
        first_store = build_store_with_backend(db_path, backend_name)
        second_store = build_store_with_backend(db_path, backend_name)
        await first_store.initialize()
        await second_store.initialize()

        assert await first_store.claim_default_memory_vector_store(namespace="default") is True
        assert await second_store.claim_default_memory_vector_store(namespace="default") is False

        await first_store.delete_default_memory_vector_store_claim(namespace="default")

        assert await second_store.claim_default_memory_vector_store(namespace="default") is True


async def test_default_memory_vector_store_claim_does_not_hide_unexpected_insert_errors(monkeypatch):
    with TemporaryDirectory() as tmpdir:
        store = build_store(f"{tmpdir}/responses.db")
        await store.initialize()

        async def fail_insert(*args, **kwargs):
            raise RuntimeError("boom")

        monkeypatch.setattr(store.sql_store, "insert", fail_insert)

        with pytest.raises(RuntimeError, match="boom"):
            await store.claim_default_memory_vector_store(namespace="default")
