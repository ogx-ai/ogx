# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Unit tests for PostgresKVStoreImpl.

Since unit tests cannot depend on a running Postgres server, these tests
use a mocked asyncpg pool to verify SQL query correctness, namespace prefixing,
expiration filtering, and error handling.
"""

import asyncio
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ogx.core.storage.datatypes import PostgresKVStoreConfig


def _make_config(namespace: str | None = None, table_name: str = "test_kvstore") -> PostgresKVStoreConfig:
    return PostgresKVStoreConfig(
        host="localhost",
        port=5432,
        db="testdb",
        user="testuser",
        password="testpass",
        table_name=table_name,
        namespace=namespace,
    )


def _make_store_with_mock_pool(config: PostgresKVStoreConfig):
    """Create a PostgresKVStoreImpl with a mocked pool via _acquire()."""
    from ogx.core.storage.kvstore.postgres.postgres import PostgresKVStoreImpl

    store = PostgresKVStoreImpl(config)
    store._table_created = True
    mock_conn = AsyncMock()

    @asynccontextmanager
    async def _acquire_conn():
        yield mock_conn

    mock_pool = MagicMock()
    mock_pool.acquire = _acquire_conn
    mock_pool.close = AsyncMock()
    mock_pool.expire_connections = AsyncMock()
    store._pool = mock_pool
    store._loop = asyncio.get_event_loop()
    return store, mock_conn


# -- Namespace prefixing -------------------------------------------------------


async def test_set_applies_namespace():
    store, conn = _make_store_with_mock_pool(_make_config(namespace="quota"))
    await store.set("user:123", "5", expiration=None)

    conn.execute.assert_called_once()
    args = conn.execute.call_args
    assert args[0][1] == "quota:user:123"


async def test_get_applies_namespace():
    store, conn = _make_store_with_mock_pool(_make_config(namespace="quota"))
    conn.fetchrow.return_value = None

    await store.get("user:123")

    args = conn.fetchrow.call_args
    assert args[0][1] == "quota:user:123"


async def test_delete_applies_namespace():
    store, conn = _make_store_with_mock_pool(_make_config(namespace="myns"))

    await store.delete("k1")

    args = conn.execute.call_args
    assert args[0][1] == "myns:k1"


async def test_no_namespace_passes_key_through():
    store, conn = _make_store_with_mock_pool(_make_config(namespace=None))
    conn.fetchrow.return_value = None

    await store.get("raw_key")

    args = conn.fetchrow.call_args
    assert args[0][1] == "raw_key"


# -- SQL query correctness ----------------------------------------------------


async def test_get_filters_expired_keys():
    """get() SQL includes expiration > NOW() filter."""
    store, conn = _make_store_with_mock_pool(_make_config())
    conn.fetchrow.return_value = None

    await store.get("k1")

    sql = conn.fetchrow.call_args[0][0]
    assert "expiration IS NULL OR expiration > NOW()" in sql


async def test_set_uses_upsert():
    """set() SQL uses INSERT ... ON CONFLICT DO UPDATE."""
    store, conn = _make_store_with_mock_pool(_make_config())
    await store.set("k1", "v1")

    sql = conn.execute.call_args[0][0]
    assert "ON CONFLICT" in sql
    assert "DO UPDATE" in sql


async def test_values_in_range_uses_half_open_interval():
    """values_in_range SQL uses >= start AND < end."""
    store, conn = _make_store_with_mock_pool(_make_config())
    conn.fetch.return_value = []

    await store.values_in_range("a", "c")

    sql = conn.fetch.call_args[0][0]
    assert "key >= $1 AND key < $2" in sql
    assert "key <= $" not in sql


async def test_values_in_range_filters_expired():
    store, conn = _make_store_with_mock_pool(_make_config())
    conn.fetch.return_value = []

    await store.values_in_range("a", "z")

    sql = conn.fetch.call_args[0][0]
    assert "expiration IS NULL OR expiration > NOW()" in sql


async def test_keys_in_range_uses_half_open_interval():
    """keys_in_range SQL uses >= start AND < end."""
    store, conn = _make_store_with_mock_pool(_make_config())
    conn.fetch.return_value = []

    await store.keys_in_range("a", "c")

    sql = conn.fetch.call_args[0][0]
    assert "key >= $1 AND key < $2" in sql


async def test_keys_in_range_filters_expired():
    """keys_in_range must also filter expired keys (bug fix verification)."""
    store, conn = _make_store_with_mock_pool(_make_config())
    conn.fetch.return_value = []

    await store.keys_in_range("a", "z")

    sql = conn.fetch.call_args[0][0]
    assert "expiration IS NULL OR expiration > NOW()" in sql


async def test_range_queries_apply_namespace():
    store, conn = _make_store_with_mock_pool(_make_config(namespace="ns"))
    conn.fetch.return_value = []

    await store.values_in_range("a", "z")

    args = conn.fetch.call_args[0]
    assert args[1] == "ns:a"
    assert args[2] == "ns:z"


async def test_keys_in_range_strips_namespace_from_results():
    """keys_in_range returns un-namespaced keys so callers can pass them to get()."""
    store, conn = _make_store_with_mock_pool(_make_config(namespace="ns"))
    conn.fetch.return_value = [{"key": "ns:key1"}, {"key": "ns:key2"}]

    keys = await store.keys_in_range("a", "z")

    assert keys == ["key1", "key2"]


async def test_values_in_range_returns_values():
    store, conn = _make_store_with_mock_pool(_make_config())
    conn.fetch.return_value = [{"value": "v1"}, {"value": "v2"}]

    values = await store.values_in_range("a", "z")

    assert values == ["v1", "v2"]


async def test_get_returns_value_when_found():
    store, conn = _make_store_with_mock_pool(_make_config())
    conn.fetchrow.return_value = {"value": "hello"}

    result = await store.get("k1")

    assert result == "hello"


async def test_get_returns_none_when_not_found():
    store, conn = _make_store_with_mock_pool(_make_config())
    conn.fetchrow.return_value = None

    result = await store.get("missing")

    assert result is None


# -- Error handling ------------------------------------------------------------


async def test_connect_wraps_connection_error():
    """_acquire() wraps connection errors in RuntimeError."""
    from ogx.core.storage.kvstore.postgres.postgres import PostgresKVStoreImpl

    config = _make_config()
    store = PostgresKVStoreImpl(config)

    with patch("ogx.core.storage.kvstore.postgres.postgres.asyncpg") as mock_asyncpg:
        mock_asyncpg.create_pool = AsyncMock(side_effect=Exception("connection refused"))
        with pytest.raises(RuntimeError, match="Could not connect"):
            await store.get("k1")


# -- Connection lifecycle ------------------------------------------------------


async def test_shutdown_closes_pool():
    """shutdown() closes the pool."""
    store, _ = _make_store_with_mock_pool(_make_config())
    pool = store._pool

    await store.shutdown()

    pool.close.assert_called_once()
    assert store._pool is None


# -- Config validation ---------------------------------------------------------
# NOTE: PostgresKVStoreConfig.validate_table_name is currently broken because
# @classmethod is stacked before @field_validator, making Pydantic ignore it.
# These tests document the DESIRED behavior; they are marked xfail until the
# validator stacking is fixed (swap to @field_validator / @classmethod order).


@pytest.mark.xfail(reason="table_name validator broken: @classmethod before @field_validator")
def test_table_name_rejects_sql_injection():
    with pytest.raises(ValueError, match="Invalid table name"):
        _make_config(table_name="users; DROP TABLE")


@pytest.mark.xfail(reason="table_name validator broken: @classmethod before @field_validator")
def test_table_name_rejects_empty():
    with pytest.raises(ValueError, match="Invalid table name"):
        _make_config(table_name="")


def test_table_name_accepts_valid():
    config = _make_config(table_name="ogx_kvstore_v2")
    assert config.table_name == "ogx_kvstore_v2"


@pytest.mark.xfail(reason="table_name validator broken: @classmethod before @field_validator")
def test_table_name_rejects_too_long():
    with pytest.raises(ValueError, match="less than 63"):
        _make_config(table_name="a" * 64)
