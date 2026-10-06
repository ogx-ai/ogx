# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Tests for upgrading a database written by a server that keyed conversation items on id alone.

The items table keeps its name, since access policies match on it. On upgrade the id-keyed
table is renamed to conversation_items_v1, a table keyed on (conversation_id, id) is created
under the old name and the rows are copied once. The v1 table is retained with a startup
warning and is never read again. The tests run on SQLite always and on PostgreSQL when
ENABLE_POSTGRES_TESTS is set.
"""

import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import pytest
from sqlalchemy import event, text
from sqlalchemy.engine import Engine

from ogx.core.conversations.conversations import (
    ITEM_KEY_COLUMNS,
    ITEMS_BACKFILL_KEY,
    ITEMS_TABLE,
    ITEMS_V1_TABLE,
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

SERVICE_TABLES = ["openai_conversations", ITEMS_TABLE, ITEMS_V1_TABLE, MIGRATIONS_TABLE]


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


V1_ITEMS_SCHEMA = {
    "id": ColumnDefinition(type=ColumnType.STRING, primary_key=True),
    "conversation_id": ColumnType.STRING,
    "created_at": ColumnType.INTEGER,
    "sort_order": ColumnType.INTEGER,
    "item_data": ColumnType.JSON,
    "owner_principal": ColumnType.STRING,
    "access_attributes": ColumnType.JSON,
    "tenant_id": ColumnType.STRING,
}


async def _v1_store(config: SqlAlchemySqlStoreConfig, table: str) -> SqlAlchemySqlStoreImpl:
    """A store with the id-keyed schema registered under ``table``; the service never registers it."""
    store = SqlAlchemySqlStoreImpl(config)
    await store.create_table(table, V1_ITEMS_SCHEMA)
    return store


async def _write_v1_database(config: SqlAlchemySqlStoreConfig, conversation_id: str) -> None:
    """Lay down the tables an earlier server would have left behind, with two items."""
    store = await _v1_store(config, ITEMS_TABLE)
    await store.create_table(
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
    await store.insert(
        "openai_conversations",
        {"id": conversation_id, "created_at": 1700000000, "metadata": None, "owner_principal": ""},
    )
    await store.insert(ITEMS_TABLE, [_v1_row(conversation_id, i) for i in range(2)])
    await store.shutdown()


def _v1_row(conversation_id: str, i: int) -> dict:
    return {
        "id": f"msg_{i}",
        "conversation_id": conversation_id,
        "created_at": 1700000000,
        "sort_order": i,
        "item_data": _message(f"v1 {i}", f"msg_{i}").model_dump(),
        "owner_principal": "alice",
        "access_attributes": {"roles": ["admin"]},
        "tenant_id": "tenant-a",
    }


@contextmanager
def _recorded_statements() -> Iterator[list[str]]:
    """Collect every SQL statement any engine executes while the block runs."""
    statements: list[str] = []

    def record(conn, cursor, statement, parameters, context, executemany) -> None:
        statements.append(statement)

    event.listen(Engine, "before_cursor_execute", record)
    try:
        yield statements
    finally:
        event.remove(Engine, "before_cursor_execute", record)


@patch("ogx.core.storage.sqlstore.authorized_sqlstore.get_authenticated_user")
async def test_initialize_migrates_id_keyed_table_in_place(mock_user, upgrade_backend):
    """The id-keyed table becomes conversation_items_v1 and its rows, with owner and tenant, are
    served from a conversation_items keyed on (conversation_id, id)."""
    conversation_id = "conv_" + "a" * 48
    await _write_v1_database(upgrade_backend, conversation_id)
    mock_user.return_value = User("alice", {"roles": ["admin"]})

    service = await _make_service(upgrade_backend)

    store = service.sql_store.sql_store
    assert await store.primary_key_columns(ITEMS_TABLE) == ITEM_KEY_COLUMNS
    assert await store.primary_key_columns(ITEMS_V1_TABLE) == ["id"]
    listed = await service.list_items(ListItemsRequest(conversation_id=conversation_id, order="asc"))
    assert _texts(listed) == [("msg_0", "v1 0"), ("msg_1", "v1 1")]
    copied = await _raw_items(service, ITEMS_TABLE, conversation_id)
    assert [
        (row["id"], row["sort_order"], row["owner_principal"], row["tenant_id"], row["access_attributes"])
        for row in copied
    ] == [
        ("msg_0", 0, "alice", "tenant-a", {"roles": ["admin"]}),
        ("msg_1", 1, "alice", "tenant-a", {"roles": ["admin"]}),
    ]
    # The v1 table is retained and a copied item can be referenced like any other.
    v1_store = await _v1_store(upgrade_backend, ITEMS_V1_TABLE)
    assert len((await v1_store.fetch_all(ITEMS_V1_TABLE)).data) == 2
    await v1_store.shutdown()
    other = await service.create_conversation(CreateConversationRequest(items=[_message("x", "msg_0")]))
    assert _texts(await service.list_items(ListItemsRequest(conversation_id=other.id))) == [("msg_0", "v1 0")]
    await store.shutdown()


async def test_second_initialize_leaves_v1_table_unread(upgrade_backend):
    """Once migrated, a boot neither renames nor copies again and never reads the v1 table."""
    conversation_id = "conv_" + "b" * 48
    await _write_v1_database(upgrade_backend, conversation_id)
    first = await _make_service(upgrade_backend)
    flag = await first.sql_store.sql_store.fetch_one(MIGRATIONS_TABLE, where={"name": ITEMS_BACKFILL_KEY})
    assert flag is not None and flag["completed_at"] > 0
    await first.sql_store.sql_store.shutdown()
    v1_store = await _v1_store(upgrade_backend, ITEMS_V1_TABLE)
    await v1_store.insert(ITEMS_V1_TABLE, _v1_row(conversation_id, 2))
    await v1_store.shutdown()

    with patch("ogx.core.conversations.conversations.logger") as logger, _recorded_statements() as statements:
        second = await _make_service(upgrade_backend)

    reads_of_v1 = [s for s in statements if f"FROM {ITEMS_V1_TABLE}" in s or f'FROM "{ITEMS_V1_TABLE}"' in s]
    assert reads_of_v1 == []
    assert not any("RENAME" in s for s in statements)
    assert await second.sql_store.sql_store.primary_key_columns(ITEMS_TABLE) == ITEM_KEY_COLUMNS
    assert [row["id"] for row in await _raw_items(second, ITEMS_TABLE, conversation_id)] == ["msg_0", "msg_1"]
    logger.info.assert_not_called()
    logger.warning.assert_called_once()
    assert logger.warning.call_args.kwargs["retained_table"] == ITEMS_V1_TABLE
    await second.sql_store.sql_store.shutdown()


async def test_fresh_database_gets_composite_key_without_migration(upgrade_backend):
    """A database with no items table gets the composite-key table directly, with no v1 table,
    no migrations table and no log line about either."""
    with patch("ogx.core.conversations.conversations.logger") as logger, _recorded_statements() as statements:
        service = await _make_service(upgrade_backend)

    store = service.sql_store.sql_store
    assert await store.primary_key_columns(ITEMS_TABLE) == ITEM_KEY_COLUMNS
    assert not await store.table_exists(ITEMS_V1_TABLE)
    assert not await store.table_exists(MIGRATIONS_TABLE)
    assert not any("RENAME" in s or f"FROM {ITEMS_V1_TABLE}" in s for s in statements)
    logger.info.assert_not_called()
    logger.warning.assert_not_called()
    await store.shutdown()
