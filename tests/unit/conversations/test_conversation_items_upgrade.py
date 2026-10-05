# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Tests for upgrading a database written by a server that keyed conversation items on id alone.

The backfill into the per-conversation table runs once per database, keeps row ownership, and
leaves the legacy table in place with a startup warning. The tests run on SQLite always and on
PostgreSQL when ENABLE_POSTGRES_TESTS is set.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from sqlalchemy import text

from ogx.core.conversations.conversations import (
    ITEMS_BACKFILL_KEY,
    ITEMS_TABLE,
    LEGACY_ITEMS_TABLE,
    MIGRATIONS_TABLE,
)
from ogx.core.datatypes import User
from ogx.core.storage.datatypes import (
    PostgresSqlStoreConfig,
    SqlAlchemySqlStoreConfig,
    SqliteSqlStoreConfig,
)
from ogx.core.storage.sqlstore.sqlalchemy_sqlstore import SqlAlchemySqlStoreImpl
from ogx_api.conversations import CreateConversationRequest, ListItemsRequest
from ogx_api.internal.sqlstore import ColumnDefinition, ColumnType
from tests.unit.conversations.test_conversations import _make_service, _message, _raw_items, _texts


def _postgres_config() -> PostgresSqlStoreConfig:
    return PostgresSqlStoreConfig(
        host=os.environ.get("POSTGRES_HOST", "localhost"),
        port=int(os.environ.get("POSTGRES_PORT", "5432")),
        db=os.environ.get("POSTGRES_DB", "ogx"),
        user=os.environ.get("POSTGRES_USER", "ogx"),
        password=os.environ.get("POSTGRES_PASSWORD", "ogx"),
    )


BACKENDS = [
    pytest.param("sqlite", id="sqlite"),
    pytest.param(
        "postgres",
        id="postgres",
        marks=pytest.mark.skipif(
            not os.environ.get("ENABLE_POSTGRES_TESTS"),
            reason="PostgreSQL tests require ENABLE_POSTGRES_TESTS environment variable",
        ),
    ),
]

SERVICE_TABLES = ["openai_conversations", LEGACY_ITEMS_TABLE, ITEMS_TABLE, MIGRATIONS_TABLE]


@pytest.fixture(params=BACKENDS)
async def upgrade_backend(request):
    """An empty backend config that outlives service instances, so a DB can be re-opened.

    Postgres keeps tables across tests, so the service tables are dropped before and after.
    """
    if request.param == "sqlite":
        with tempfile.TemporaryDirectory() as tmpdir:
            yield SqliteSqlStoreConfig(db_path=str(Path(tmpdir) / "upgrade.db"))
        return
    config = _postgres_config()
    await _drop_tables(config, SERVICE_TABLES)
    yield config
    await _drop_tables(config, SERVICE_TABLES)


async def _drop_tables(config: SqlAlchemySqlStoreConfig, tables: list[str]) -> None:
    engine = SqlAlchemySqlStoreImpl(config).create_engine()
    async with engine.begin() as conn:
        for table in tables:
            await conn.execute(text(f'DROP TABLE IF EXISTS "{table}"'))
    await engine.dispose()


LEGACY_ITEMS_SCHEMA = {
    "id": ColumnDefinition(type=ColumnType.STRING, primary_key=True),
    "conversation_id": ColumnType.STRING,
    "created_at": ColumnType.INTEGER,
    "sort_order": ColumnType.INTEGER,
    "item_data": ColumnType.JSON,
    "owner_principal": ColumnType.STRING,
    "access_attributes": ColumnType.JSON,
    "tenant_id": ColumnType.STRING,
}


async def _legacy_store(config: SqlAlchemySqlStoreConfig) -> SqlAlchemySqlStoreImpl:
    """A store with the id-keyed legacy table registered; the service itself never registers it."""
    store = SqlAlchemySqlStoreImpl(config)
    await store.create_table(LEGACY_ITEMS_TABLE, LEGACY_ITEMS_SCHEMA)
    return store


async def _write_legacy_database(config: SqlAlchemySqlStoreConfig, conversation_id: str) -> None:
    """Lay down the tables an earlier server would have left behind, with two items."""
    legacy_store = await _legacy_store(config)
    await legacy_store.create_table(
        "openai_conversations",
        {
            "id": ColumnDefinition(type=ColumnType.STRING, primary_key=True),
            "created_at": ColumnType.INTEGER,
            "items": ColumnType.JSON,
            "metadata": ColumnType.JSON,
            "owner_principal": ColumnType.STRING,
            "access_attributes": ColumnType.JSON,
        },
    )
    await legacy_store.insert(
        "openai_conversations",
        {"id": conversation_id, "created_at": 1700000000, "metadata": None, "owner_principal": ""},
    )
    await legacy_store.insert(LEGACY_ITEMS_TABLE, [_legacy_row(conversation_id, i) for i in range(2)])
    await legacy_store.shutdown()


def _legacy_row(conversation_id: str, i: int) -> dict:
    return {
        "id": f"msg_{i}",
        "conversation_id": conversation_id,
        "created_at": 1700000000,
        "sort_order": i,
        "item_data": _message(f"legacy {i}", f"msg_{i}").model_dump(),
        "owner_principal": "alice",
        "access_attributes": {"roles": ["admin"]},
        "tenant_id": "tenant-a",
    }


@patch("ogx.core.storage.sqlstore.authorized_sqlstore.get_authenticated_user")
async def test_initialize_backfills_legacy_items_with_owner_and_tenant(mock_user, upgrade_backend):
    """Rows written by an earlier server are readable by their owner after upgrade and keep owner and tenant."""
    conversation_id = "conv_" + "a" * 48
    await _write_legacy_database(upgrade_backend, conversation_id)
    mock_user.return_value = User("alice", {"roles": ["admin"]})

    service = await _make_service(upgrade_backend)

    listed = await service.list_items(ListItemsRequest(conversation_id=conversation_id, order="asc"))
    assert _texts(listed) == [("msg_0", "legacy 0"), ("msg_1", "legacy 1")]
    copied = await _raw_items(service, ITEMS_TABLE, conversation_id)
    assert [(row["id"], row["owner_principal"], row["tenant_id"], row["access_attributes"]) for row in copied] == [
        ("msg_0", "alice", "tenant-a", {"roles": ["admin"]}),
        ("msg_1", "alice", "tenant-a", {"roles": ["admin"]}),
    ]
    # The legacy table is retained and a copied item can be referenced like any other.
    legacy_store = await _legacy_store(upgrade_backend)
    assert len((await legacy_store.fetch_all(LEGACY_ITEMS_TABLE)).data) == 2
    await legacy_store.shutdown()
    other = await service.create_conversation(CreateConversationRequest(items=[_message("x", "msg_0")]))
    assert _texts(await service.list_items(ListItemsRequest(conversation_id=other.id))) == [("msg_0", "legacy 0")]
    await service.sql_store.sql_store.shutdown()


async def test_initialize_skips_backfill_once_flag_is_set(upgrade_backend):
    """A second initialize against the same database must not copy again, even if the legacy table grew."""
    conversation_id = "conv_" + "b" * 48
    await _write_legacy_database(upgrade_backend, conversation_id)
    first = await _make_service(upgrade_backend)
    flag = await first.sql_store.sql_store.fetch_one(MIGRATIONS_TABLE, where={"name": ITEMS_BACKFILL_KEY})
    assert flag is not None and flag["completed_at"] > 0
    await first.sql_store.sql_store.shutdown()
    legacy_store = await _legacy_store(upgrade_backend)
    await legacy_store.insert(LEGACY_ITEMS_TABLE, _legacy_row(conversation_id, 2))
    await legacy_store.shutdown()

    with patch("ogx.core.conversations.conversations.logger") as logger:
        second = await _make_service(upgrade_backend)

    assert [row["id"] for row in await _raw_items(second, ITEMS_TABLE, conversation_id)] == ["msg_0", "msg_1"]
    logger.info.assert_not_called()
    logger.warning.assert_called_once()
    assert logger.warning.call_args.kwargs["legacy_table"] == LEGACY_ITEMS_TABLE
    await second.sql_store.sql_store.shutdown()


async def test_initialize_on_fresh_database_does_not_warn_about_legacy_table():
    with (
        tempfile.TemporaryDirectory() as tmpdir,
        patch("ogx.core.conversations.conversations.logger") as logger,
    ):
        service = await _make_service(SqliteSqlStoreConfig(db_path=str(Path(tmpdir) / "fresh.db")))
        await service.sql_store.sql_store.shutdown()

    logger.warning.assert_not_called()
    assert logger.info.call_args.kwargs["copied_rows"] == 0
