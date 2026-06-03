# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Tests that SqlAlchemySqlStoreImpl works correctly across event loop boundaries."""

import asyncio

import pytest

from ogx.core.storage.datatypes import SqliteSqlStoreConfig
from ogx.core.storage.sqlstore.sqlalchemy_sqlstore import SqlAlchemySqlStoreImpl
from ogx_api.internal.sqlstore import ColumnDefinition, ColumnType


@pytest.fixture
def db_path(tmp_path):
    return str(tmp_path / "test_loop.db")


def test_reset_engine_clears_state(db_path):
    """reset_engine() sets engine and session to None."""
    store = SqlAlchemySqlStoreImpl(SqliteSqlStoreConfig(db_path=db_path))

    async def init():
        await store.create_table("t", {"id": ColumnDefinition(type=ColumnType.STRING, primary_key=True)})
        await store.insert("t", {"id": "1"})

    asyncio.run(init())
    assert store._engine is not None
    assert store.async_session is not None

    store.reset_engine()
    assert store._engine is None
    assert store.async_session is None


def test_reset_engine_preserves_metadata(db_path):
    """reset_engine() preserves table metadata so _ensure_engine recreates tables."""
    store = SqlAlchemySqlStoreImpl(SqliteSqlStoreConfig(db_path=db_path))

    async def init():
        await store.create_table("t", {"id": ColumnDefinition(type=ColumnType.STRING, primary_key=True)})

    asyncio.run(init())
    assert "t" in store.metadata.tables

    store.reset_engine()
    assert "t" in store.metadata.tables


def test_data_survives_engine_reset(db_path):
    """Data written before reset_engine() is accessible after engine recreation."""
    store = SqlAlchemySqlStoreImpl(SqliteSqlStoreConfig(db_path=db_path))

    async def phase1():
        await store.create_table(
            "items",
            {
                "id": ColumnDefinition(type=ColumnType.STRING, primary_key=True),
                "value": ColumnType.STRING,
            },
        )
        await store.insert("items", {"id": "row1", "value": "before_reset"})

    asyncio.run(phase1())
    store.reset_engine()

    async def phase2():
        await store.insert("items", {"id": "row2", "value": "after_reset"})
        result = await store.fetch_all("items")
        return result

    result = asyncio.run(phase2())
    ids = {row["id"] for row in result.data}
    assert ids == {"row1", "row2"}


def test_engine_reset_across_event_loops(db_path):
    """Simulates the server init pattern: init in one loop, use in another."""
    store = SqlAlchemySqlStoreImpl(SqliteSqlStoreConfig(db_path=db_path))

    async def init_in_temp_loop():
        await store.create_table(
            "items",
            {
                "id": ColumnDefinition(type=ColumnType.STRING, primary_key=True),
                "value": ColumnType.STRING,
            },
        )
        await store.insert("items", {"id": "init_row", "value": "from_init"})
        result = await store.fetch_all("items")
        assert len(result.data) == 1

    asyncio.run(init_in_temp_loop())

    store.reset_engine()

    async def use_in_request_loop():
        await store.insert("items", {"id": "request_row", "value": "from_request"})
        result = await store.fetch_all("items")
        return result

    result = asyncio.run(use_in_request_loop())
    assert len(result.data) == 2
    ids = {row["id"] for row in result.data}
    assert ids == {"init_row", "request_row"}
