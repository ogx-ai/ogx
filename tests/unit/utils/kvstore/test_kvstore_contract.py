# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Backend-contract tests for KVStore implementations.

Runs the same test suite against every backend that can be instantiated
without external services (InmemoryKVStoreImpl, SqliteKVStoreImpl).
Every backend must produce identical results for these tests.
"""

from datetime import UTC, datetime, timedelta

import pytest

from ogx.core.storage.datatypes import SqliteKVStoreConfig
from ogx.core.storage.kvstore.kvstore import InmemoryKVStoreImpl
from ogx.core.storage.kvstore.sqlite.sqlite import SqliteKVStoreImpl
from ogx_api.internal.kvstore import KVStore


@pytest.fixture
async def inmemory_store():
    store = InmemoryKVStoreImpl()
    await store.initialize()
    yield store
    await store.shutdown()


@pytest.fixture
async def sqlite_store():
    config = SqliteKVStoreConfig(db_path=":memory:")
    store = SqliteKVStoreImpl(config)
    await store.initialize()
    yield store
    await store.shutdown()


@pytest.fixture(params=["inmemory", "sqlite"])
async def store(request, inmemory_store, sqlite_store) -> KVStore:
    if request.param == "inmemory":
        return inmemory_store
    return sqlite_store


# -- Basic CRUD ---------------------------------------------------------------


async def test_get_nonexistent_returns_none(store: KVStore):
    assert await store.get("does_not_exist") is None


async def test_set_and_get(store: KVStore):
    await store.set("k1", "v1")
    assert await store.get("k1") == "v1"


async def test_overwrite(store: KVStore):
    await store.set("k1", "first")
    await store.set("k1", "second")
    assert await store.get("k1") == "second"


async def test_delete(store: KVStore):
    await store.set("k1", "v1")
    await store.delete("k1")
    assert await store.get("k1") is None


async def test_delete_nonexistent_is_noop(store: KVStore):
    await store.delete("never_existed")


# -- Range queries (half-open interval) ---------------------------------------


async def test_values_in_range_half_open(store: KVStore):
    """end_key is excluded (half-open interval [start, end))."""
    await store.set("a", "va")
    await store.set("b", "vb")
    await store.set("c", "vc")

    result = await store.values_in_range("a", "c")
    assert result == ["va", "vb"]


async def test_keys_in_range_half_open(store: KVStore):
    """end_key is excluded (half-open interval [start, end))."""
    await store.set("a", "va")
    await store.set("b", "vb")
    await store.set("c", "vc")

    result = await store.keys_in_range("a", "c")
    assert result == ["a", "b"]


async def test_range_includes_start_key(store: KVStore):
    await store.set("a", "va")
    await store.set("b", "vb")

    result = await store.values_in_range("a", "z")
    assert "va" in result


async def test_range_empty_when_no_keys_match(store: KVStore):
    await store.set("a", "va")

    assert await store.values_in_range("m", "z") == []
    assert await store.keys_in_range("m", "z") == []


async def test_range_ordering(store: KVStore):
    """Results are returned in sorted key order."""
    await store.set("c", "vc")
    await store.set("a", "va")
    await store.set("b", "vb")

    values = await store.values_in_range("a", "d")
    assert values == ["va", "vb", "vc"]

    keys = await store.keys_in_range("a", "d")
    assert keys == sorted(keys)


async def test_range_single_key(store: KVStore):
    """Range containing exactly one key."""
    await store.set("b", "vb")
    await store.set("a", "va")
    await store.set("c", "vc")

    assert await store.values_in_range("b", "c") == ["vb"]


# -- Expiration ---------------------------------------------------------------


async def test_expired_key_not_returned_by_get(store: KVStore):
    past = datetime.now(tz=UTC) - timedelta(seconds=1)
    await store.set("k1", "v1", expiration=past)
    assert await store.get("k1") is None


async def test_non_expired_key_returned_by_get(store: KVStore):
    future = datetime.now(tz=UTC) + timedelta(hours=1)
    await store.set("k1", "v1", expiration=future)
    assert await store.get("k1") == "v1"


async def test_no_expiration_never_expires(store: KVStore):
    await store.set("k1", "v1", expiration=None)
    assert await store.get("k1") == "v1"


async def test_expired_keys_excluded_from_values_in_range(store: KVStore):
    past = datetime.now(tz=UTC) - timedelta(seconds=1)
    future = datetime.now(tz=UTC) + timedelta(hours=1)

    await store.set("a", "va", expiration=past)
    await store.set("b", "vb", expiration=future)
    await store.set("c", "vc")

    result = await store.values_in_range("a", "d")
    assert result == ["vb", "vc"]


async def test_expired_keys_excluded_from_keys_in_range(store: KVStore):
    past = datetime.now(tz=UTC) - timedelta(seconds=1)
    future = datetime.now(tz=UTC) + timedelta(hours=1)

    await store.set("a", "va", expiration=past)
    await store.set("b", "vb", expiration=future)
    await store.set("c", "vc")

    keys = await store.keys_in_range("a", "d")
    assert "a" not in keys
    assert len(keys) == 2


# -- keys_in_range → get roundtrip ---------------------------------------------


async def test_keys_from_range_can_be_passed_to_get(store: KVStore):
    """Keys returned by keys_in_range must work as arguments to get()."""
    await store.set("item:1", "v1")
    await store.set("item:2", "v2")

    keys = await store.keys_in_range("item:", "item:\xff")
    assert len(keys) == 2

    for key in keys:
        value = await store.get(key)
        assert value is not None


async def test_keys_from_range_can_be_passed_to_delete(store: KVStore):
    """Keys returned by keys_in_range must work as arguments to delete()."""
    await store.set("del:1", "v1")
    await store.set("del:2", "v2")

    keys = await store.keys_in_range("del:", "del:\xff")
    for key in keys:
        await store.delete(key)

    assert await store.get("del:1") is None
    assert await store.get("del:2") is None


# -- Shutdown -----------------------------------------------------------------


async def test_shutdown_idempotent(store: KVStore):
    await store.shutdown()
    await store.shutdown()


# -- Namespace isolation (separate instances) ----------------------------------


async def test_namespace_isolation_inmemory():
    """Keys in different namespaces do not interfere."""
    store_a = InmemoryKVStoreImpl(namespace="ns_a")
    store_b = InmemoryKVStoreImpl(namespace="ns_b")
    await store_a.initialize()
    await store_b.initialize()

    await store_a.set("key", "value_a")
    await store_b.set("key", "value_b")

    assert await store_a.get("key") == "value_a"
    assert await store_b.get("key") == "value_b"

    await store_a.delete("key")
    assert await store_a.get("key") is None
    assert await store_b.get("key") == "value_b"

    await store_a.shutdown()
    await store_b.shutdown()


async def test_namespace_isolation_sqlite(tmp_path):
    """Keys in different namespaces do not interfere (shared db)."""
    db_path = str(tmp_path / "test.db")
    config_a = SqliteKVStoreConfig(db_path=db_path, namespace="ns_a")
    config_b = SqliteKVStoreConfig(db_path=db_path, namespace="ns_b")

    store_a = SqliteKVStoreImpl(config_a)
    store_b = SqliteKVStoreImpl(config_b)
    await store_a.initialize()
    await store_b.initialize()

    await store_a.set("key", "value_a")
    await store_b.set("key", "value_b")

    assert await store_a.get("key") == "value_a"
    assert await store_b.get("key") == "value_b"

    await store_a.delete("key")
    assert await store_a.get("key") is None
    assert await store_b.get("key") == "value_b"

    await store_a.shutdown()
    await store_b.shutdown()


async def test_namespace_range_isolation_sqlite(tmp_path):
    """Range queries only return keys from the same namespace."""
    db_path = str(tmp_path / "test.db")
    config_a = SqliteKVStoreConfig(db_path=db_path, namespace="ns_a")
    config_b = SqliteKVStoreConfig(db_path=db_path, namespace="ns_b")

    store_a = SqliteKVStoreImpl(config_a)
    store_b = SqliteKVStoreImpl(config_b)
    await store_a.initialize()
    await store_b.initialize()

    await store_a.set("a", "a_in_ns_a")
    await store_a.set("b", "b_in_ns_a")
    await store_b.set("a", "a_in_ns_b")

    values_a = await store_a.values_in_range("a", "z")
    values_b = await store_b.values_in_range("a", "z")

    assert values_a == ["a_in_ns_a", "b_in_ns_a"]
    assert values_b == ["a_in_ns_b"]

    await store_a.shutdown()
    await store_b.shutdown()
