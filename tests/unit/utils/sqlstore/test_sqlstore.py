# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import sys
import time
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pytest

from ogx.core.storage.datatypes import PostgresSqlStoreConfig
from ogx.core.storage.sqlstore.sqlalchemy_sqlstore import SqlAlchemySqlStoreImpl
from ogx.core.storage.sqlstore.sqlstore import SqliteSqlStoreConfig

_SQLSTORE_MODULE = sys.modules[SqlAlchemySqlStoreImpl.__module__]
from ogx_api.internal.sqlstore import ColumnDefinition, ColumnType


async def test_sqlstore_shutdown_disposes_engine():
    """Test that shutdown() properly disposes the async engine.

    This is critical for aiosqlite >= 0.22 where worker threads are non-daemon.
    Without proper engine disposal, the process hangs on exit.
    See: https://github.com/ogx-ai/ogx/issues/4587
    """
    with TemporaryDirectory() as tmp_dir:
        db_path = tmp_dir + "/shutdown_test.db"
        store = SqlAlchemySqlStoreImpl(SqliteSqlStoreConfig(db_path=db_path))

        # Create a table and insert data to ensure connections are established
        # (this triggers lazy engine initialization)
        await store.create_table(
            "test",
            {"id": ColumnType.INTEGER, "name": ColumnType.STRING},
        )
        await store.insert("test", {"id": 1, "name": "test"})

        # Verify engine exists (after lazy init)
        assert store._engine is not None

        # Shutdown should dispose the engine and close connections
        await store.shutdown()

        # Engine should be None after shutdown
        assert store._engine is None, (
            "Engine not disposed after shutdown. This causes process hang on exit with aiosqlite >= 0.22"
        )


async def test_sqlite_sqlstore():
    with TemporaryDirectory() as tmp_dir:
        db_name = "test.db"
        sqlstore = SqlAlchemySqlStoreImpl(
            SqliteSqlStoreConfig(
                db_path=tmp_dir + "/" + db_name,
            )
        )
        await sqlstore.create_table(
            table="test",
            schema={
                "id": ColumnType.INTEGER,
                "name": ColumnType.STRING,
            },
        )
        await sqlstore.insert("test", {"id": 1, "name": "test"})
        await sqlstore.insert("test", {"id": 12, "name": "test12"})
        result = await sqlstore.fetch_all("test")
        assert result.data == [{"id": 1, "name": "test"}, {"id": 12, "name": "test12"}]
        assert result.has_more is False

        row = await sqlstore.fetch_one("test", {"id": 1})
        assert row == {"id": 1, "name": "test"}

        row = await sqlstore.fetch_one("test", {"name": "test12"})
        assert row == {"id": 12, "name": "test12"}

        # order by
        result = await sqlstore.fetch_all("test", order_by=[("id", "asc")])
        assert result.data == [{"id": 1, "name": "test"}, {"id": 12, "name": "test12"}]

        result = await sqlstore.fetch_all("test", order_by=[("id", "desc")])
        assert result.data == [{"id": 12, "name": "test12"}, {"id": 1, "name": "test"}]

        # limit
        result = await sqlstore.fetch_all("test", limit=1)
        assert result.data == [{"id": 1, "name": "test"}]
        assert result.has_more is True

        # update
        await sqlstore.update("test", {"name": "test123"}, {"id": 1})
        row = await sqlstore.fetch_one("test", {"id": 1})
        assert row == {"id": 1, "name": "test123"}

        # delete
        await sqlstore.delete("test", {"id": 1})
        result = await sqlstore.fetch_all("test")
        assert result.data == [{"id": 12, "name": "test12"}]
        assert result.has_more is False


async def test_sqlstore_upsert_support():
    with TemporaryDirectory() as tmp_dir:
        db_path = tmp_dir + "/upsert.db"
        store = SqlAlchemySqlStoreImpl(SqliteSqlStoreConfig(db_path=db_path))

        await store.create_table(
            "items",
            {
                "id": ColumnDefinition(type=ColumnType.STRING, primary_key=True),
                "value": ColumnType.STRING,
                "updated_at": ColumnType.INTEGER,
            },
        )

        await store.upsert(
            table="items",
            data={"id": "item_1", "value": "first", "updated_at": 1},
            conflict_columns=["id"],
        )
        row = await store.fetch_one("items", {"id": "item_1"})
        assert row == {"id": "item_1", "value": "first", "updated_at": 1}

        await store.upsert(
            table="items",
            data={"id": "item_1", "value": "second", "updated_at": 2},
            conflict_columns=["id"],
            update_columns=["value", "updated_at"],
        )
        row = await store.fetch_one("items", {"id": "item_1"})
        assert row == {"id": "item_1", "value": "second", "updated_at": 2}


async def test_sqlstore_pagination_basic():
    """Test basic pagination functionality at the SQL store level."""
    with TemporaryDirectory() as tmp_dir:
        db_path = tmp_dir + "/test.db"
        store = SqlAlchemySqlStoreImpl(SqliteSqlStoreConfig(db_path=db_path))

        # Create test table
        await store.create_table(
            "test_records",
            {
                "id": ColumnType.STRING,
                "created_at": ColumnType.INTEGER,
                "name": ColumnType.STRING,
            },
        )

        # Insert test data
        base_time = int(time.time())
        test_data = [
            {"id": "zebra", "created_at": base_time + 1, "name": "First"},
            {"id": "apple", "created_at": base_time + 2, "name": "Second"},
            {"id": "moon", "created_at": base_time + 3, "name": "Third"},
            {"id": "banana", "created_at": base_time + 4, "name": "Fourth"},
            {"id": "car", "created_at": base_time + 5, "name": "Fifth"},
        ]

        for record in test_data:
            await store.insert("test_records", record)

        # Test 1: First page (no cursor)
        result = await store.fetch_all(
            table="test_records",
            order_by=[("created_at", "desc")],
            limit=2,
        )
        assert len(result.data) == 2
        assert result.data[0]["id"] == "car"  # Most recent first
        assert result.data[1]["id"] == "banana"
        assert result.has_more is True

        # Test 2: Second page using cursor
        result2 = await store.fetch_all(
            table="test_records",
            order_by=[("created_at", "desc")],
            cursor=("id", "banana"),
            limit=2,
        )
        assert len(result2.data) == 2
        assert result2.data[0]["id"] == "moon"
        assert result2.data[1]["id"] == "apple"
        assert result2.has_more is True

        # Test 3: Final page
        result3 = await store.fetch_all(
            table="test_records",
            order_by=[("created_at", "desc")],
            cursor=("id", "apple"),
            limit=2,
        )
        assert len(result3.data) == 1
        assert result3.data[0]["id"] == "zebra"
        assert result3.has_more is False


async def test_sqlstore_pagination_with_filter():
    """Test pagination with WHERE conditions."""
    with TemporaryDirectory() as tmp_dir:
        db_path = tmp_dir + "/test.db"
        store = SqlAlchemySqlStoreImpl(SqliteSqlStoreConfig(db_path=db_path))

        # Create test table
        await store.create_table(
            "test_records",
            {
                "id": ColumnType.STRING,
                "created_at": ColumnType.INTEGER,
                "category": ColumnType.STRING,
            },
        )

        # Insert test data with categories
        base_time = int(time.time())
        test_data = [
            {"id": "xyz", "created_at": base_time + 1, "category": "A"},
            {"id": "def", "created_at": base_time + 2, "category": "B"},
            {"id": "pqr", "created_at": base_time + 3, "category": "A"},
            {"id": "abc", "created_at": base_time + 4, "category": "B"},
        ]

        for record in test_data:
            await store.insert("test_records", record)

        # Test pagination with filter
        result = await store.fetch_all(
            table="test_records",
            where={"category": "A"},
            order_by=[("created_at", "desc")],
            limit=1,
        )
        assert len(result.data) == 1
        assert result.data[0]["id"] == "pqr"  # Most recent category A
        assert result.has_more is True

        # Second page with filter
        result2 = await store.fetch_all(
            table="test_records",
            where={"category": "A"},
            order_by=[("created_at", "desc")],
            cursor=("id", "pqr"),
            limit=1,
        )
        assert len(result2.data) == 1
        assert result2.data[0]["id"] == "xyz"
        assert result2.has_more is False


async def test_sqlstore_pagination_ascending_order():
    """Test pagination with ascending order."""
    with TemporaryDirectory() as tmp_dir:
        db_path = tmp_dir + "/test.db"
        store = SqlAlchemySqlStoreImpl(SqliteSqlStoreConfig(db_path=db_path))

        # Create test table
        await store.create_table(
            "test_records",
            {
                "id": ColumnType.STRING,
                "created_at": ColumnType.INTEGER,
            },
        )

        # Insert test data
        base_time = int(time.time())
        test_data = [
            {"id": "gamma", "created_at": base_time + 1},
            {"id": "alpha", "created_at": base_time + 2},
            {"id": "beta", "created_at": base_time + 3},
        ]

        for record in test_data:
            await store.insert("test_records", record)

        # Test ascending order
        result = await store.fetch_all(
            table="test_records",
            order_by=[("created_at", "asc")],
            limit=1,
        )
        assert len(result.data) == 1
        assert result.data[0]["id"] == "gamma"  # Oldest first
        assert result.has_more is True

        # Second page with ascending order
        result2 = await store.fetch_all(
            table="test_records",
            order_by=[("created_at", "asc")],
            cursor=("id", "gamma"),
            limit=1,
        )
        assert len(result2.data) == 1
        assert result2.data[0]["id"] == "alpha"
        assert result2.has_more is True


async def test_sqlstore_pagination_multi_column_ordering_error():
    """Test that multi-column ordering raises an error when using cursor pagination."""
    with TemporaryDirectory() as tmp_dir:
        db_path = tmp_dir + "/test.db"
        store = SqlAlchemySqlStoreImpl(SqliteSqlStoreConfig(db_path=db_path))

        # Create test table
        await store.create_table(
            "test_records",
            {
                "id": ColumnType.STRING,
                "priority": ColumnType.INTEGER,
                "created_at": ColumnType.INTEGER,
            },
        )

        await store.insert("test_records", {"id": "task1", "priority": 1, "created_at": 12345})

        # Test that multi-column ordering with cursor raises error
        with pytest.raises(ValueError, match="Cursor pagination only supports single-column ordering, got 2 columns"):
            await store.fetch_all(
                table="test_records",
                order_by=[("priority", "asc"), ("created_at", "desc")],
                cursor=("id", "task1"),
                limit=2,
            )

        # Test that multi-column ordering without cursor works fine
        result = await store.fetch_all(
            table="test_records",
            order_by=[("priority", "asc"), ("created_at", "desc")],
            limit=2,
        )
        assert len(result.data) == 1
        assert result.data[0]["id"] == "task1"


async def test_sqlstore_pagination_cursor_requires_order_by():
    """Test that cursor pagination requires order_by parameter."""
    with TemporaryDirectory() as tmp_dir:
        db_path = tmp_dir + "/test.db"
        store = SqlAlchemySqlStoreImpl(SqliteSqlStoreConfig(db_path=db_path))

        await store.create_table("test_records", {"id": ColumnType.STRING})
        await store.insert("test_records", {"id": "task1"})

        # Test that cursor without order_by raises error
        with pytest.raises(ValueError, match="order_by is required when using cursor pagination"):
            await store.fetch_all(
                table="test_records",
                cursor=("id", "task1"),
            )


async def test_sqlstore_pagination_error_handling():
    """Test error handling for invalid columns and cursor IDs."""
    with TemporaryDirectory() as tmp_dir:
        db_path = tmp_dir + "/test.db"
        store = SqlAlchemySqlStoreImpl(SqliteSqlStoreConfig(db_path=db_path))

        # Create test table
        await store.create_table(
            "test_records",
            {
                "id": ColumnType.STRING,
                "name": ColumnType.STRING,
            },
        )

        await store.insert("test_records", {"id": "test1", "name": "Test"})

        # Test invalid cursor tuple format
        with pytest.raises(ValueError, match="Cursor must be a tuple of"):
            await store.fetch_all(
                table="test_records",
                order_by=[("name", "asc")],
                cursor="invalid",  # Should be tuple
            )

        # Test invalid cursor_key_column
        with pytest.raises(ValueError, match="Cursor key column 'nonexistent' not found in table"):
            await store.fetch_all(
                table="test_records",
                order_by=[("name", "asc")],
                cursor=("nonexistent", "test1"),
            )

        # Test invalid order_by column
        with pytest.raises(ValueError, match="Column 'invalid_col' not found in table"):
            await store.fetch_all(
                table="test_records",
                order_by=[("invalid_col", "asc")],
            )

        # Test nonexistent cursor_id
        with pytest.raises(ValueError, match="Record with id='nonexistent' not found in table"):
            await store.fetch_all(
                table="test_records",
                order_by=[("name", "asc")],
                cursor=("id", "nonexistent"),
            )


async def test_where_operator_gt_and_update_delete():
    with TemporaryDirectory() as tmp_dir:
        db_path = tmp_dir + "/test.db"
        store = SqlAlchemySqlStoreImpl(SqliteSqlStoreConfig(db_path=db_path))

        await store.create_table(
            "items",
            {
                "id": ColumnType.INTEGER,
                "value": ColumnType.INTEGER,
                "name": ColumnType.STRING,
            },
        )

        await store.insert("items", {"id": 1, "value": 10, "name": "one"})
        await store.insert("items", {"id": 2, "value": 20, "name": "two"})
        await store.insert("items", {"id": 3, "value": 30, "name": "three"})

        result = await store.fetch_all("items", where={"value": {">": 15}})
        assert {r["id"] for r in result.data} == {2, 3}

        row = await store.fetch_one("items", where={"value": {">=": 30}})
        assert row["id"] == 3

        await store.update("items", {"name": "small"}, {"value": {"<": 25}})
        rows = (await store.fetch_all("items")).data
        names = {r["id"]: r["name"] for r in rows}
        assert names[1] == "small"
        assert names[2] == "small"
        assert names[3] == "three"

        await store.delete("items", {"id": {"==": 2}})
        rows_after = (await store.fetch_all("items")).data
        assert {r["id"] for r in rows_after} == {1, 3}


async def test_batch_insert():
    with TemporaryDirectory() as tmp_dir:
        db_path = tmp_dir + "/test.db"
        store = SqlAlchemySqlStoreImpl(SqliteSqlStoreConfig(db_path=db_path))

        await store.create_table(
            "batch_test",
            {
                "id": ColumnType.INTEGER,
                "name": ColumnType.STRING,
                "value": ColumnType.INTEGER,
            },
        )

        batch_data = [
            {"id": 1, "name": "first", "value": 10},
            {"id": 2, "name": "second", "value": 20},
            {"id": 3, "name": "third", "value": 30},
        ]

        await store.insert("batch_test", batch_data)

        result = await store.fetch_all("batch_test", order_by=[("id", "asc")])
        assert result.data == batch_data


async def test_where_operator_edge_cases():
    with TemporaryDirectory() as tmp_dir:
        db_path = tmp_dir + "/test.db"
        store = SqlAlchemySqlStoreImpl(SqliteSqlStoreConfig(db_path=db_path))

        await store.create_table(
            "events",
            {"id": ColumnType.STRING, "ts": ColumnType.INTEGER},
        )

        base = 1024
        await store.insert("events", {"id": "a", "ts": base - 10})
        await store.insert("events", {"id": "b", "ts": base + 10})

        row = await store.fetch_one("events", where={"id": "a"})
        assert row["id"] == "a"

        with pytest.raises(ValueError, match="Unsupported operator"):
            await store.fetch_all("events", where={"ts": {"!=": base}})


async def test_sqlstore_pagination_custom_key_column():
    """Test pagination with custom primary key column (not 'id')."""
    with TemporaryDirectory() as tmp_dir:
        db_path = tmp_dir + "/test.db"
        store = SqlAlchemySqlStoreImpl(SqliteSqlStoreConfig(db_path=db_path))

        # Create test table with custom primary key
        await store.create_table(
            "custom_table",
            {
                "uuid": ColumnType.STRING,
                "timestamp": ColumnType.INTEGER,
                "data": ColumnType.STRING,
            },
        )

        # Insert test data
        base_time = int(time.time())
        test_data = [
            {"uuid": "uuid-alpha", "timestamp": base_time + 1, "data": "First"},
            {"uuid": "uuid-beta", "timestamp": base_time + 2, "data": "Second"},
            {"uuid": "uuid-gamma", "timestamp": base_time + 3, "data": "Third"},
        ]

        for record in test_data:
            await store.insert("custom_table", record)

        # Test pagination with custom key column
        result = await store.fetch_all(
            table="custom_table",
            order_by=[("timestamp", "desc")],
            limit=2,
        )
        assert len(result.data) == 2
        assert result.data[0]["uuid"] == "uuid-gamma"  # Most recent
        assert result.data[1]["uuid"] == "uuid-beta"
        assert result.has_more is True

        # Second page using custom key column
        result2 = await store.fetch_all(
            table="custom_table",
            order_by=[("timestamp", "desc")],
            cursor=("uuid", "uuid-beta"),  # Use uuid as key column
            limit=2,
        )
        assert len(result2.data) == 1
        assert result2.data[0]["uuid"] == "uuid-alpha"
        assert result2.has_more is False


@pytest.mark.parametrize("pre_ping", [True, False])
async def test_pool_pre_ping_propagates_to_engine(pre_ping):
    """pool_pre_ping config value is forwarded to create_async_engine."""
    with patch.object(_SQLSTORE_MODULE, "create_async_engine") as mock_create:
        config = PostgresSqlStoreConfig(user="test", password="test", pool_pre_ping=pre_ping)
        store = SqlAlchemySqlStoreImpl(config)
        await store._ensure_engine()
        mock_create.assert_called_once()
        _, kwargs = mock_create.call_args
        assert kwargs["pool_pre_ping"] is pre_ping


async def test_pool_pre_ping_defaults_to_true():
    """Both SQLite and Postgres configs default pool_pre_ping to True."""
    sqlite_cfg = SqliteSqlStoreConfig(db_path="/tmp/test.db")
    assert sqlite_cfg.pool_pre_ping is True

    pg_cfg = PostgresSqlStoreConfig(user="test", password="test")
    assert pg_cfg.pool_pre_ping is True


async def test_postgres_pool_config_defaults():
    """PostgresSqlStoreConfig exposes pool tuning knobs with sensible defaults."""
    cfg = PostgresSqlStoreConfig(user="test", password="test")
    assert cfg.pool_size == 10
    assert cfg.max_overflow == 20
    assert cfg.pool_recycle == -1


async def test_postgres_pool_kwargs_propagate_to_engine():
    """Postgres pool_size, max_overflow, and pool_recycle are forwarded to create_async_engine."""
    with patch.object(_SQLSTORE_MODULE, "create_async_engine") as mock_create:
        config = PostgresSqlStoreConfig(
            user="test",
            password="test",
            pool_size=15,
            max_overflow=25,
            pool_recycle=300,
        )
        store = SqlAlchemySqlStoreImpl(config)
        await store._ensure_engine()
        _, kwargs = mock_create.call_args
        assert kwargs["pool_size"] == 15
        assert kwargs["max_overflow"] == 25
        assert kwargs["pool_recycle"] == 300


async def test_postgres_pool_recycle_omitted_when_disabled():
    """pool_recycle kwarg is not passed to create_async_engine when set to -1."""
    with patch.object(_SQLSTORE_MODULE, "create_async_engine") as mock_create:
        config = PostgresSqlStoreConfig(
            user="test",
            password="test",
            pool_recycle=-1,
        )
        store = SqlAlchemySqlStoreImpl(config)
        await store._ensure_engine()
        _, kwargs = mock_create.call_args
        assert "pool_recycle" not in kwargs


async def test_pool_recycle_is_configurable():
    """pool_recycle interval can be customized."""
    cfg = PostgresSqlStoreConfig(
        user="test",
        password="test",
        pool_recycle=300,
    )
    assert cfg.pool_recycle == 300
