# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from tempfile import TemporaryDirectory
from uuid import uuid4

from ogx.core.storage.datatypes import ResponsesStoreReference, SqliteSqlStoreConfig
from ogx.core.storage.sqlstore.sqlstore import register_sqlstore_backends
from ogx.providers.utils.responses.responses_store import ResponsesStore


def build_store(db_path: str) -> ResponsesStore:
    backend_name = f"sql_responses_memory_{uuid4().hex}"
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
