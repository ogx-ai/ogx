# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from ogx.core.access_control.datatypes import Action
from ogx.core.datatypes import TenancyConfig, TenancyMode
from ogx.core.storage.sqlstore.authorized_sqlstore import set_default_tenancy_config
from ogx.providers.utils.memory.openai_vector_store_mixin import (
    OPENAI_VECTOR_STORES_FILE_BATCHES_PREFIX,
    OPENAI_VECTOR_STORES_FILES_CONTENTS_PREFIX,
    OPENAI_VECTOR_STORES_FILES_PREFIX,
    OPENAI_VECTOR_STORES_PREFIX,
    OPENAI_VECTOR_STORES_SQL_MIGRATION_KEY,
    TABLE_VECTOR_STORE_FILE_CONTENTS,
    TABLE_VECTOR_STORE_FILES,
    OpenAIVectorStoreMixin,
)
from ogx_api import (
    OpenAIUpdateVectorStoreRequest,
    VectorStoreChunkingStrategyAuto,
    VectorStoreNotFoundError,
)
from ogx_api.vector_io.models import OpenAIAttachFileRequest


def _make_store_info():
    """Build a minimal in-memory vector store dict matching the mixin's expectations."""
    return {
        "file_ids": [],
        "file_counts": {"total": 0, "completed": 0, "cancelled": 0, "failed": 0, "in_progress": 0},
        "metadata": {},
    }


def _make_full_store_info(store_id: str, name: str) -> dict:
    return {
        "id": store_id,
        "object": "vector_store",
        "created_at": 1,
        "name": name,
        "usage_bytes": 0,
        "file_counts": {"total": 0, "completed": 0, "cancelled": 0, "failed": 0, "in_progress": 0},
        "status": "completed",
        "expires_after": None,
        "expires_at": None,
        "last_active_at": 1,
        "metadata": {},
        "file_ids": [],
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

    async def test_migration_stamps_single_tenant_on_raw_sql_rows(self):
        set_default_tenancy_config(TenancyConfig(mode=TenancyMode.SINGLE, default_tenant_id="default-tenant"))
        try:
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
                    "tenant_id": "default-tenant",
                },
                conflict_columns=["id"],
                update_columns=["store_data", "tenant_id"],
            )
        finally:
            set_default_tenancy_config(TenancyConfig())

    async def test_migration_requires_default_tenant_for_multi_tenancy(self):
        set_default_tenancy_config(TenancyConfig(mode=TenancyMode.MULTI))
        try:
            kvstore = self._make_kvstore({f"{OPENAI_VECTOR_STORES_PREFIX}vs_abc": json.dumps({"id": "vs_abc"})})
            sql_store = self._make_sql_store()
            metadata_store = self._make_metadata_store(sql_store)

            mixin = MockVectorStoreMixin(
                inference_api=AsyncMock(),
                files_api=AsyncMock(),
                kvstore=kvstore,
                metadata_store=metadata_store,
            )
            mixin.metadata_store = metadata_store

            with pytest.raises(ValueError, match="Failed to migrate vector store metadata"):
                await mixin._migrate_kvstore_to_sql()
            sql_store.upsert.assert_not_called()
        finally:
            set_default_tenancy_config(TenancyConfig())

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


class TestMetadataStoreEnforcement:
    """Tests for mandatory metadata_store when access control policies are active."""

    async def test_initialize_raises_when_policy_set_but_no_metadata_store(self):
        mixin = MockVectorStoreMixin(
            inference_api=AsyncMock(),
            files_api=AsyncMock(),
            kvstore=AsyncMock(),
        )
        mixin._policy = [MagicMock()]

        with pytest.raises(ValueError, match="metadata_store is required"):
            await mixin.initialize_openai_vector_stores()

    async def test_initialize_raises_when_multi_tenancy_set_but_no_metadata_store(self):
        set_default_tenancy_config(TenancyConfig(mode=TenancyMode.MULTI))
        try:
            mixin = MockVectorStoreMixin(
                inference_api=AsyncMock(),
                files_api=AsyncMock(),
                kvstore=AsyncMock(),
            )
            mixin._policy = []

            with pytest.raises(ValueError, match="tenancy mode is 'multi'"):
                await mixin.initialize_openai_vector_stores()
        finally:
            set_default_tenancy_config(TenancyConfig())

    async def test_initialize_succeeds_when_no_policy(self):
        kvstore = AsyncMock()
        kvstore.values_in_range = AsyncMock(return_value=[])

        mixin = MockVectorStoreMixin(
            inference_api=AsyncMock(),
            files_api=AsyncMock(),
            kvstore=kvstore,
        )
        mixin._policy = []

        await mixin.initialize_openai_vector_stores()

    async def test_openai_vector_store_api_uses_authorized_metadata_store(self):
        allowed = _make_full_store_info("vs_allowed", "allowed")
        hidden = _make_full_store_info("vs_hidden", "hidden")

        metadata_store = MagicMock()
        metadata_store.fetch_all = AsyncMock(return_value=MagicMock(data=[{"store_data": allowed}]))

        async def _fetch_one(table, where, action=Action.READ):
            if where["id"] == "vs_allowed":
                return {"store_data": allowed}
            return None

        metadata_store.fetch_one = AsyncMock(side_effect=_fetch_one)
        metadata_store.update = AsyncMock()
        metadata_store.delete = AsyncMock()

        mixin = MockVectorStoreMixin(
            inference_api=AsyncMock(),
            files_api=AsyncMock(),
            metadata_store=metadata_store,
        )
        mixin.openai_vector_stores = {
            "vs_allowed": allowed,
            "vs_hidden": hidden,
        }

        listed = await mixin.openai_list_vector_stores()
        assert [store.id for store in listed.data] == ["vs_allowed"]

        retrieved = await mixin.openai_retrieve_vector_store("vs_allowed")
        assert retrieved.id == "vs_allowed"

        with pytest.raises(VectorStoreNotFoundError):
            await mixin.openai_retrieve_vector_store("vs_hidden")
        with pytest.raises(VectorStoreNotFoundError):
            await mixin.openai_update_vector_store("vs_hidden", OpenAIUpdateVectorStoreRequest(name="blocked"))
        with pytest.raises(VectorStoreNotFoundError):
            await mixin.openai_delete_vector_store("vs_hidden")

        assert "vs_hidden" in mixin.openai_vector_stores
        metadata_store.update.assert_not_called()
        metadata_store.delete.assert_not_called()

    async def test_delete_vector_store_file_uses_delete_authorization_only(self):
        store_info = _make_full_store_info("vs_allowed", "allowed")
        store_info["file_ids"] = ["file_1"]
        store_info["file_counts"] = {
            "total": 1,
            "completed": 1,
            "cancelled": 0,
            "failed": 0,
            "in_progress": 0,
        }
        file_info = {
            "id": "file_1",
            "object": "vector_store.file",
            "usage_bytes": 100,
            "created_at": 1,
            "vector_store_id": "vs_allowed",
            "status": "completed",
            "chunking_strategy": {"type": "auto"},
        }
        actions = []

        async def _fetch_one(table, where, action=Action.READ):
            actions.append(action)
            if action == Action.DELETE:
                return {"store_data": store_info}
            return None

        async def _raw_fetch_all(table, **kwargs):
            if table == TABLE_VECTOR_STORE_FILES:
                return MagicMock(data=[{"file_data": file_info}])
            if table == TABLE_VECTOR_STORE_FILE_CONTENTS:
                return MagicMock(data=[])
            return MagicMock(data=[])

        metadata_store = MagicMock()
        metadata_store.fetch_one = AsyncMock(side_effect=_fetch_one)
        metadata_store.sql_store = MagicMock()
        metadata_store.sql_store.fetch_all = AsyncMock(side_effect=_raw_fetch_all)
        metadata_store.delete = AsyncMock()
        metadata_store.upsert = AsyncMock()

        mixin = MockVectorStoreMixin(
            inference_api=AsyncMock(),
            files_api=AsyncMock(),
            metadata_store=metadata_store,
        )

        result = await mixin.openai_delete_vector_store_file("vs_allowed", "file_1")

        assert result.deleted is True
        assert actions == [Action.DELETE]
        assert mixin.openai_vector_stores["vs_allowed"]["file_ids"] == []
        metadata_store.delete.assert_awaited()
        metadata_store.upsert.assert_awaited()

    async def test_initialize_succeeds_when_policy_and_metadata_store_set(self):
        metadata_store = MagicMock()
        metadata_store.create_table = AsyncMock()
        metadata_store.sql_store = AsyncMock()
        metadata_store.sql_store.fetch_all = AsyncMock(return_value=MagicMock(data=[]))
        metadata_store.fetch_all = AsyncMock(return_value=MagicMock(data=[]))

        kvstore = AsyncMock()
        kvstore.get = AsyncMock(return_value="1")
        kvstore.values_in_range = AsyncMock(return_value=[])

        mixin = MockVectorStoreMixin(
            inference_api=AsyncMock(),
            files_api=AsyncMock(),
            kvstore=kvstore,
            metadata_store=metadata_store,
        )
        mixin._policy = [MagicMock()]

        await mixin.initialize_openai_vector_stores()


class TestFileBatchCleanup:
    """Tests for metadata-store file batch cleanup behavior."""

    async def test_cleanup_uses_unfiltered_sql_store_access(self):
        sql_store = AsyncMock()
        sql_store.fetch_all = AsyncMock(
            return_value=MagicMock(
                data=[
                    {
                        "id": "batch_1",
                        "batch_data": {"id": "batch_1", "expires_at": 1},
                    }
                ]
            )
        )
        sql_store.delete = AsyncMock()

        metadata_store = MagicMock()
        metadata_store.sql_store = sql_store
        metadata_store.fetch_all = AsyncMock(side_effect=AssertionError("filtered fetch_all should not be used"))

        mixin = MockVectorStoreMixin(
            inference_api=AsyncMock(),
            files_api=AsyncMock(),
            kvstore=AsyncMock(),
            metadata_store=metadata_store,
        )
        mixin.openai_file_batches = {"batch_1": {"id": "batch_1"}}

        await mixin._cleanup_expired_file_batches()

        metadata_store.fetch_all.assert_not_called()
        sql_store.fetch_all.assert_called_once_with(table="vector_store_file_batches")
        sql_store.delete.assert_called_once_with(
            table="vector_store_file_batches",
            where={"id": "batch_1"},
        )
        assert "batch_1" not in mixin.openai_file_batches
