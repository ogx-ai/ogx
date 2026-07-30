# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Unit tests for KVStore delete behavior across backends."""

import tempfile

from ogx.core.storage.datatypes import SqliteKVStoreConfig
from ogx.core.storage.kvstore.kvstore import InmemoryKVStoreImpl
from ogx.core.storage.kvstore.sqlite.sqlite import SqliteKVStoreImpl


class TestKVStoreDelete:
    """All KVStore backends should behave consistently on delete."""

    async def test_inmemory_delete_existing_key(self):
        store = InmemoryKVStoreImpl()
        await store.initialize()

        await store.set("k", "v")
        assert await store.get("k") == "v"

        await store.delete("k")
        assert await store.get("k") is None

    async def test_inmemory_delete_missing_key(self):
        store = InmemoryKVStoreImpl()
        await store.initialize()

        await store.delete("nonexistent")

    async def test_sqlite_delete_existing_key(self):
        config = SqliteKVStoreConfig(db_path=":memory:")
        store = SqliteKVStoreImpl(config)
        await store.initialize()

        await store.set("k", "v")
        assert await store.get("k") == "v"

        await store.delete("k")
        assert await store.get("k") is None

        await store.shutdown()

    async def test_sqlite_delete_missing_key(self):
        config = SqliteKVStoreConfig(db_path=":memory:")
        store = SqliteKVStoreImpl(config)
        await store.initialize()

        await store.delete("nonexistent")

        await store.shutdown()

    async def test_delete_idempotent(self):
        """Deleting the same key twice should not raise."""
        store = InmemoryKVStoreImpl()
        await store.initialize()

        await store.set("k", "v")
        await store.delete("k")
        await store.delete("k")

        assert await store.get("k") is None

    async def test_delete_does_not_affect_other_keys(self):
        store = InmemoryKVStoreImpl()
        await store.initialize()

        await store.set("a", "1")
        await store.set("b", "2")

        await store.delete("a")

        assert await store.get("a") is None
        assert await store.get("b") == "2"

    async def test_set_after_delete(self):
        """A deleted key can be re-set."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = SqliteKVStoreConfig(db_path=f"{tmpdir}/test.db")
            store = SqliteKVStoreImpl(config)
            await store.initialize()

            await store.set("k", "v1")
            await store.delete("k")
            await store.set("k", "v2")

            assert await store.get("k") == "v2"

            await store.shutdown()
