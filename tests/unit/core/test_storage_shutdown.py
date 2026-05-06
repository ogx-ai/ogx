# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Unit tests for storage backend shutdown functionality."""

import tempfile

from ogx.core.storage.datatypes import (
    KVStoreReference,
    SqliteKVStoreConfig,
    SqliteSqlStoreConfig,
    SqlStoreReference,
)
from ogx.core.storage.kvstore.kvstore import (
    InmemoryKVStoreImpl,
    kvstore_impl,
    register_kvstore_backends,
    shutdown_kvstore_backends,
)
from ogx.core.storage.kvstore.sqlite.sqlite import SqliteKVStoreImpl
from ogx.core.storage.sqlstore.sqlalchemy_sqlstore import SqlAlchemySqlStoreImpl
from ogx.core.storage.sqlstore.sqlstore import (
    register_sqlstore_backends,
    shutdown_sqlstore_backends,
    sqlstore_impl,
)


class TestKVStoreShutdown:
    """Tests for KV store shutdown functionality."""

    async def test_sqlite_kvstore_shutdown_memory(self):
        """Test that SqliteKVStoreImpl properly shuts down in-memory connections."""
        config = SqliteKVStoreConfig(db_path=":memory:")
        store = SqliteKVStoreImpl(config)
        await store.initialize()

        # Verify connection is open
        assert store._conn is not None

        # Set some data
        await store.set("test_key", "test_value")
        value = await store.get("test_key")
        assert value == "test_value"

        # Shutdown
        await store.shutdown()

        # Verify connection is closed
        assert store._conn is None

    async def test_sqlite_kvstore_shutdown_file(self):
        """Test that SqliteKVStoreImpl properly shuts down file-based connections."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = SqliteKVStoreConfig(db_path=f"{tmpdir}/test.db")
            store = SqliteKVStoreImpl(config)
            await store.initialize()

            # File-based stores don't keep persistent connections
            assert store._conn is None

            # Set some data (uses connection-per-operation)
            await store.set("test_key", "test_value")
            value = await store.get("test_key")
            assert value == "test_value"

            # Shutdown should complete without error
            await store.shutdown()

    async def test_inmemory_kvstore_shutdown(self):
        """Test that InmemoryKVStoreImpl properly shuts down."""
        store = InmemoryKVStoreImpl()
        await store.initialize()

        # Set some data
        await store.set("test_key", "test_value")
        value = await store.get("test_key")
        assert value == "test_value"

        # Shutdown clears the store
        await store.shutdown()

        # Store should be empty after shutdown
        assert len(store._store) == 0

    async def test_shutdown_kvstore_backends(self):
        """Test that shutdown_kvstore_backends shuts down all registered instances."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Register backends
            register_kvstore_backends(
                {
                    "kv_test": SqliteKVStoreConfig(db_path=f"{tmpdir}/kv.db"),
                }
            )

            # Create instances by calling kvstore_impl with KVStoreReference
            store1 = await kvstore_impl(KVStoreReference(backend="kv_test", namespace="ns1"))
            store2 = await kvstore_impl(KVStoreReference(backend="kv_test", namespace="ns2"))

            # Verify stores are working
            await store1.set("key1", "value1")
            await store2.set("key2", "value2")

            # Shutdown all backends
            await shutdown_kvstore_backends()

            # After shutdown, the instances cache should be cleared
            # (we can't easily verify the stores are closed without accessing internals)


class TestSqlStoreShutdown:
    """Tests for SQL store shutdown functionality."""

    async def test_sqlalchemy_sqlstore_shutdown(self):
        """Test that SqlAlchemySqlStoreImpl properly disposes the engine."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = SqliteSqlStoreConfig(db_path=f"{tmpdir}/test.db")
            store = SqlAlchemySqlStoreImpl(config)

            # Create a table and insert data (this triggers lazy engine initialization)
            from ogx_api.internal.sqlstore import ColumnType

            await store.create_table("test", {"id": ColumnType.INTEGER, "name": ColumnType.STRING})
            await store.insert("test", {"id": 1, "name": "test"})

            # Verify session maker has an engine (after lazy init)
            assert store.async_session is not None
            engine = store.async_session.kw.get("bind")
            assert engine is not None

            # Shutdown
            await store.shutdown()

            # Verify shutdown was called (engine should be disposed but async_session still exists)
            # We can't easily verify the engine is disposed without accessing internal state,
            # but we can verify shutdown doesn't throw and can be called

    async def test_shutdown_sqlstore_backends(self):
        """Test that shutdown_sqlstore_backends shuts down all registered instances."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Register backends
            register_sqlstore_backends(
                {
                    "sql_test": SqliteSqlStoreConfig(db_path=f"{tmpdir}/sql.db"),
                }
            )

            # Create instance by calling sqlstore_impl
            store = sqlstore_impl(SqlStoreReference(backend="sql_test", table_name="test"))

            # Verify store is working
            from ogx_api.internal.sqlstore import ColumnType

            await store.create_table("test", {"id": ColumnType.INTEGER})
            await store.insert("test", {"id": 1})

            # Shutdown all backends
            await shutdown_sqlstore_backends()


class TestKVStoreProtocolShutdown:
    """Tests to verify all KVStore implementations have shutdown method."""

    async def test_sqlite_kvstore_has_shutdown(self):
        """Verify SqliteKVStoreImpl has shutdown method."""
        config = SqliteKVStoreConfig(db_path=":memory:")
        store = SqliteKVStoreImpl(config)
        assert hasattr(store, "shutdown")
        assert callable(store.shutdown)

    async def test_inmemory_kvstore_has_shutdown(self):
        """Verify InmemoryKVStoreImpl has shutdown method."""
        store = InmemoryKVStoreImpl()
        assert hasattr(store, "shutdown")
        assert callable(store.shutdown)


class TestSqlStoreProtocolShutdown:
    """Tests to verify SqlStore implementations have shutdown method."""

    async def test_sqlalchemy_sqlstore_has_shutdown(self):
        """Verify SqlAlchemySqlStoreImpl has shutdown method."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = SqliteSqlStoreConfig(db_path=f"{tmpdir}/test.db")
            store = SqlAlchemySqlStoreImpl(config)
            assert hasattr(store, "shutdown")
            assert callable(store.shutdown)
            # Clean up
            await store.shutdown()


class TestShutdownIdempotency:
    """Tests for shutdown idempotency (calling shutdown multiple times)."""

    async def test_sqlite_kvstore_shutdown_idempotent(self):
        """Test that SqliteKVStoreImpl.shutdown() can be called multiple times."""
        config = SqliteKVStoreConfig(db_path=":memory:")
        store = SqliteKVStoreImpl(config)
        await store.initialize()

        # Shutdown multiple times should not raise
        await store.shutdown()
        await store.shutdown()
        await store.shutdown()

    async def test_sqlalchemy_sqlstore_shutdown_idempotent(self):
        """Test that SqlAlchemySqlStoreImpl.shutdown() can be called multiple times."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = SqliteSqlStoreConfig(db_path=f"{tmpdir}/test.db")
            store = SqlAlchemySqlStoreImpl(config)

            # Shutdown multiple times should not raise
            await store.shutdown()
            await store.shutdown()
            await store.shutdown()

    async def test_inmemory_kvstore_shutdown_idempotent(self):
        """Test that InmemoryKVStoreImpl.shutdown() can be called multiple times."""
        store = InmemoryKVStoreImpl()
        await store.initialize()

        # Shutdown multiple times should not raise
        await store.shutdown()
        await store.shutdown()
        await store.shutdown()
