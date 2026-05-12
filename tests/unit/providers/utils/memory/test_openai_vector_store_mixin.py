# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from ogx.providers.utils.memory.openai_vector_store_mixin import (
    OPENAI_VECTOR_STORES_FILE_BATCHES_PREFIX,
    OPENAI_VECTOR_STORES_FILES_CONTENTS_PREFIX,
    OPENAI_VECTOR_STORES_FILES_PREFIX,
    OPENAI_VECTOR_STORES_PREFIX,
    OPENAI_VECTOR_STORES_SQL_MIGRATION_KEY,
    OpenAIVectorStoreMixin,
)
from ogx_api import (
    VectorStoreChunkingStrategyAuto,
)
from ogx_api.vector_io.models import OpenAIAttachFileRequest


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


class TestOpenAIVectorStoreMixin:
    """Unit tests for OpenAIVectorStoreMixin."""

    @pytest.fixture
    def mock_files_api(self):
        mock = AsyncMock()
        mock.openai_retrieve_file = AsyncMock()
        mock.openai_retrieve_file.return_value = MagicMock(filename="test.pdf")
        return mock

    @pytest.fixture
    def mock_inference_api(self):
        return AsyncMock()

    @pytest.fixture
    def mock_kvstore(self):
        kv = AsyncMock()
        kv.set = AsyncMock()
        kv.get = AsyncMock(return_value=None)
        return kv

    async def test_missing_file_processor_api_returns_failed_status(
        self, mock_inference_api, mock_files_api, mock_kvstore
    ):
        """Test that missing file_processor_api marks the file as failed with a clear error."""
        mixin = MockVectorStoreMixin(
            inference_api=mock_inference_api,
            files_api=mock_files_api,
            kvstore=mock_kvstore,
            file_processor_api=None,
        )

        vector_store_id = "test_vector_store"
        file_id = "test_file_id"
        mixin.openai_vector_stores[vector_store_id] = _make_store_info()

        result = await mixin.openai_attach_file_to_vector_store(
            vector_store_id=vector_store_id,
            request=OpenAIAttachFileRequest(
                file_id=file_id,
                chunking_strategy=VectorStoreChunkingStrategyAuto(),
            ),
        )

        assert result.status == "failed"
        assert result.last_error is not None
        assert "FileProcessor API is required" in result.last_error.message

    async def test_file_processor_api_configured_succeeds(self, mock_inference_api, mock_files_api, mock_kvstore):
        """Test that with file_processor_api configured, processing proceeds past the check."""
        mock_file_processor_api = AsyncMock()
        mock_file_processor_api.process_file = AsyncMock()
        mock_file_processor_api.process_file.return_value = MagicMock(chunks=[], metadata={"processor": "pypdf"})

        mixin = MockVectorStoreMixin(
            inference_api=mock_inference_api,
            files_api=mock_files_api,
            kvstore=mock_kvstore,
            file_processor_api=mock_file_processor_api,
        )

        vector_store_id = "test_vector_store"
        file_id = "test_file_id"
        mixin.openai_vector_stores[vector_store_id] = _make_store_info()

        result = await mixin.openai_attach_file_to_vector_store(
            vector_store_id=vector_store_id,
            request=OpenAIAttachFileRequest(
                file_id=file_id,
                chunking_strategy=VectorStoreChunkingStrategyAuto(),
            ),
        )

        # Should not fail with the file_processor_api error
        if result.last_error:
            assert "FileProcessor API is required" not in result.last_error.message


class TestKVStoreToSQLMigration:
    """Tests for automatic KVStore-to-SQL migration during initialization."""

    def _make_kvstore(self, data: dict[str, str]) -> AsyncMock:
        """Build a mock KVStore populated with the given key-value pairs."""
        kv = AsyncMock()

        async def _set(key: str, value: str, expiration=None) -> None:
            data[key] = value

        kv.set = AsyncMock(side_effect=_set)
        kv.get = AsyncMock(side_effect=lambda key: data.get(key))

        def _values_in_range(start: str, end: str) -> list[str]:
            return [v for k, v in sorted(data.items()) if start <= k < end]

        def _keys_in_range(start: str, end: str) -> list[str]:
            return [k for k in sorted(data.keys()) if start <= k < end]

        kv.values_in_range = AsyncMock(side_effect=_values_in_range)
        kv.keys_in_range = AsyncMock(side_effect=_keys_in_range)
        return kv

    def _make_sql_store(self, existing_rows: list | None = None) -> AsyncMock:
        """Build a mock SqlStore backing the AuthorizedSqlStore."""
        sql = AsyncMock()
        fetch_result = MagicMock()
        fetch_result.data = existing_rows or []
        sql.fetch_all = AsyncMock(return_value=fetch_result)
        sql.upsert = AsyncMock()
        return sql

    def _make_metadata_store(self, sql_store: AsyncMock) -> MagicMock:
        """Build a mock AuthorizedSqlStore wrapping the given SqlStore."""
        meta = MagicMock()
        meta.sql_store = sql_store
        meta.create_table = AsyncMock()
        meta.fetch_all = AsyncMock(return_value=MagicMock(data=[]))
        return meta

    async def test_migration_copies_stores_from_kvstore_to_sql(self):
        store_info = {"id": "vs_abc", "name": "test", "status": "completed"}
        kv_data = {f"{OPENAI_VECTOR_STORES_PREFIX}vs_abc": json.dumps(store_info)}

        kvstore = self._make_kvstore(kv_data)
        sql_store = self._make_sql_store()
        metadata_store = self._make_metadata_store(sql_store)

        mixin = MockVectorStoreMixin(
            inference_api=AsyncMock(),
            files_api=AsyncMock(),
            kvstore=kvstore,
            metadata_store=metadata_store,
        )
        mixin.metadata_store = metadata_store

        await mixin._migrate_kvstore_to_sql()

        sql_store.upsert.assert_any_call(
            table="vector_stores",
            data={
                "id": "vs_abc",
                "store_data": store_info,
                "owner_principal": "",
                "access_attributes": None,
            },
            conflict_columns=["id"],
            update_columns=["store_data"],
        )

    async def test_migration_skipped_when_migration_marker_exists(self):
        kvstore = self._make_kvstore(
            {
                f"{OPENAI_VECTOR_STORES_PREFIX}vs_abc": json.dumps({"id": "vs_abc"}),
                OPENAI_VECTOR_STORES_SQL_MIGRATION_KEY: "1",
            }
        )
        sql_store = self._make_sql_store()
        metadata_store = self._make_metadata_store(sql_store)

        mixin = MockVectorStoreMixin(
            inference_api=AsyncMock(),
            files_api=AsyncMock(),
            kvstore=kvstore,
            metadata_store=metadata_store,
        )
        mixin.metadata_store = metadata_store

        await mixin._migrate_kvstore_to_sql()

        sql_store.upsert.assert_not_called()

    async def test_migration_skipped_when_kvstore_is_empty(self):
        kvstore = self._make_kvstore({})
        sql_store = self._make_sql_store()
        metadata_store = self._make_metadata_store(sql_store)

        mixin = MockVectorStoreMixin(
            inference_api=AsyncMock(),
            files_api=AsyncMock(),
            kvstore=kvstore,
            metadata_store=metadata_store,
        )
        mixin.metadata_store = metadata_store

        await mixin._migrate_kvstore_to_sql()

        sql_store.upsert.assert_not_called()
        kvstore.set.assert_any_call(key=OPENAI_VECTOR_STORES_SQL_MIGRATION_KEY, value="1")

    async def test_migration_copies_files_and_chunks(self):
        store_info = {"id": "vs_1", "name": "s", "status": "completed"}
        file_info = {"id": "file_a", "status": "completed"}
        chunk_0 = {"content": "hello"}
        chunk_1 = {"content": "world"}

        kv_data = {
            f"{OPENAI_VECTOR_STORES_PREFIX}vs_1": json.dumps(store_info),
            f"{OPENAI_VECTOR_STORES_FILES_PREFIX}vs_1:file_a": json.dumps(file_info),
            f"{OPENAI_VECTOR_STORES_FILES_CONTENTS_PREFIX}vs_1:file_a:0": json.dumps(chunk_0),
            f"{OPENAI_VECTOR_STORES_FILES_CONTENTS_PREFIX}vs_1:file_a:1": json.dumps(chunk_1),
        }

        kvstore = self._make_kvstore(kv_data)
        sql_store = self._make_sql_store()
        metadata_store = self._make_metadata_store(sql_store)

        mixin = MockVectorStoreMixin(
            inference_api=AsyncMock(),
            files_api=AsyncMock(),
            kvstore=kvstore,
            metadata_store=metadata_store,
        )
        mixin.metadata_store = metadata_store

        await mixin._migrate_kvstore_to_sql()

        assert sql_store.upsert.call_count == 4  # 1 store + 1 file + 2 chunks

        sql_store.upsert.assert_any_call(
            table="vector_store_files",
            data={
                "id": "vs_1:file_a",
                "store_id": "vs_1",
                "file_id": "file_a",
                "file_data": file_info,
                "owner_principal": "",
                "access_attributes": None,
            },
            conflict_columns=["id"],
            update_columns=["store_id", "file_id", "file_data"],
        )

    async def test_migration_copies_batches(self):
        store_info = {"id": "vs_1", "name": "s", "status": "completed"}
        batch_info = {"id": "batch_1", "vector_store_id": "vs_1", "expires_at": 99}

        kv_data = {
            f"{OPENAI_VECTOR_STORES_PREFIX}vs_1": json.dumps(store_info),
            f"{OPENAI_VECTOR_STORES_FILE_BATCHES_PREFIX}batch_1": json.dumps(batch_info),
        }

        kvstore = self._make_kvstore(kv_data)
        sql_store = self._make_sql_store()
        metadata_store = self._make_metadata_store(sql_store)

        mixin = MockVectorStoreMixin(
            inference_api=AsyncMock(),
            files_api=AsyncMock(),
            kvstore=kvstore,
            metadata_store=metadata_store,
        )
        mixin.metadata_store = metadata_store

        await mixin._migrate_kvstore_to_sql()

        sql_store.upsert.assert_any_call(
            table="vector_store_file_batches",
            data={
                "id": "batch_1",
                "store_id": "vs_1",
                "batch_data": batch_info,
                "expires_at": 99,
                "owner_principal": "",
                "access_attributes": None,
            },
            conflict_columns=["id"],
            update_columns=["store_id", "batch_data", "expires_at"],
        )
