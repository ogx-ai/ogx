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

import asyncio
import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import pytest
from sqlalchemy import event, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import DBAPIError

from ogx.core.access_control.datatypes import AccessRule, Action, Scope
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

# The composite-keyed items table the service creates, with every column the backfill copies,
# so a crashed migration can be left with a target that a later boot can finish filling.
COMPOSITE_ITEMS_SCHEMA = {
    "conversation_id": ColumnDefinition(type=ColumnType.STRING, primary_key=True),
    "id": ColumnDefinition(type=ColumnType.STRING, primary_key=True),
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


async def _write_v1_database(
    config: SqlAlchemySqlStoreConfig,
    conversation_id: str,
    owner_principal: str = "",
) -> None:
    """Lay down the tables an earlier server would have left behind, with two items.

    ``owner_principal`` is the owner stored on the conversation row; the items are always
    owned by alice (see ``_v1_row``). An empty owner leaves the conversation unowned.
    """
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
        {"id": conversation_id, "created_at": 1700000000, "metadata": None, "owner_principal": owner_principal},
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


# --- Crash-recovery -----------------------------------------------------------
#
# Each helper leaves the database in the state a hard crash would leave it: the migration's
# rename, recreate and backfill are separate transactions, so any of them can be the last one
# that completes. A later boot must recognize the state and finish where it stopped.


async def _renamed_not_copied(config: SqlAlchemySqlStoreConfig, conversation_id: str) -> None:
    """The id-keyed table is renamed to v1 but the composite table is not yet created: a crash
    between the rename and the recreate leaves the live table missing entirely."""
    await _write_v1_database(config, conversation_id)
    store = SqlAlchemySqlStoreImpl(config)
    await store.rename_table(ITEMS_TABLE, ITEMS_V1_TABLE)
    await store.shutdown()


async def _created_not_copied(config: SqlAlchemySqlStoreConfig, conversation_id: str) -> None:
    """The composite table is created (empty) but the rows have not yet been copied: a crash after
    the recreate and before the backfill."""
    await _renamed_not_copied(config, conversation_id)
    store = SqlAlchemySqlStoreImpl(config)
    await store.create_table(ITEMS_TABLE, COMPOSITE_ITEMS_SCHEMA)
    await store.shutdown()


async def _copied_not_flagged(config: SqlAlchemySqlStoreConfig, conversation_id: str) -> None:
    """Some rows are copied but the completion flag is not yet set: a crash mid-backfill. The
    anti-join top-up must fill only what is missing on the next boot, with no duplicates."""
    await _created_not_copied(config, conversation_id)
    store = SqlAlchemySqlStoreImpl(config)
    await store.create_table(ITEMS_TABLE, COMPOSITE_ITEMS_SCHEMA)
    await store.insert(ITEMS_TABLE, _v1_row(conversation_id, 0))
    await store.shutdown()


@patch("ogx.core.storage.sqlstore.authorized_sqlstore.get_authenticated_user")
async def test_recovers_when_renamed_but_not_copied(mock_user, upgrade_backend):
    """A crash between the rename and the recreate leaves the live table missing; the next boot
    recreates it, copies the rows with owner and tenant preserved, and sets the flag."""
    conversation_id = "conv_" + "c" * 48
    await _renamed_not_copied(upgrade_backend, conversation_id)

    before = SqlAlchemySqlStoreImpl(upgrade_backend)
    assert not await before.table_exists(ITEMS_TABLE)
    assert await before.table_exists(ITEMS_V1_TABLE)
    await before.shutdown()

    mock_user.return_value = User("alice", {"roles": ["admin"]})
    service = await _make_service(upgrade_backend)
    store = service.sql_store.sql_store
    assert await store.primary_key_columns(ITEMS_TABLE) == ITEM_KEY_COLUMNS
    flag = await store.fetch_one(MIGRATIONS_TABLE, where={"name": ITEMS_BACKFILL_KEY})
    assert flag is not None and flag["completed_at"] > 0
    copied = await _raw_items(service, ITEMS_TABLE, conversation_id)
    assert [(row["id"], row["owner_principal"], row["tenant_id"]) for row in copied] == [
        ("msg_0", "alice", "tenant-a"),
        ("msg_1", "alice", "tenant-a"),
    ]
    await store.shutdown()


@patch("ogx.core.storage.sqlstore.authorized_sqlstore.get_authenticated_user")
async def test_recovers_when_created_but_not_copied(mock_user, upgrade_backend):
    """A crash after the recreate and before the backfill leaves an empty composite table; the next
    boot copies the rows and sets the flag without renaming, since the key is already composite."""
    conversation_id = "conv_" + "d" * 48
    await _created_not_copied(upgrade_backend, conversation_id)
    mock_user.return_value = User("alice", {"roles": ["admin"]})

    with _recorded_statements() as statements:
        service = await _make_service(upgrade_backend)
    assert not any("RENAME" in s for s in statements)

    store = service.sql_store.sql_store
    assert await store.primary_key_columns(ITEMS_TABLE) == ITEM_KEY_COLUMNS
    flag = await store.fetch_one(MIGRATIONS_TABLE, where={"name": ITEMS_BACKFILL_KEY})
    assert flag is not None and flag["completed_at"] > 0
    copied = await _raw_items(service, ITEMS_TABLE, conversation_id)
    assert [row["id"] for row in copied] == ["msg_0", "msg_1"]
    await store.shutdown()


@patch("ogx.core.storage.sqlstore.authorized_sqlstore.get_authenticated_user")
async def test_recovers_when_copied_but_not_flagged(mock_user, upgrade_backend):
    """A crash mid-backfill leaves some rows copied but no flag; the next boot tops up only the
    missing rows and sets the flag, with no duplicates and ownership preserved."""
    conversation_id = "conv_" + "e" * 48
    await _copied_not_flagged(upgrade_backend, conversation_id)
    mock_user.return_value = User("alice", {"roles": ["admin"]})

    service = await _make_service(upgrade_backend)
    store = service.sql_store.sql_store
    assert await store.primary_key_columns(ITEMS_TABLE) == ITEM_KEY_COLUMNS
    flag = await store.fetch_one(MIGRATIONS_TABLE, where={"name": ITEMS_BACKFILL_KEY})
    assert flag is not None and flag["completed_at"] > 0
    copied = await _raw_items(service, ITEMS_TABLE, conversation_id)
    assert [(row["id"], row["owner_principal"], row["tenant_id"]) for row in copied] == [
        ("msg_0", "alice", "tenant-a"),
        ("msg_1", "alice", "tenant-a"),
    ]
    await store.shutdown()


# --- Access-policy compatibility ----------------------------------------------
#
# The live table keeps its name precisely so that access policies naming it keep matching.
# These tests drive reads through the authorized store with policies whose resource globs
# reference conversation_items and confirm the migrated rows are still what the policies see.


def _items_policy(resource: str, *, owner_scoped: bool = False) -> list[AccessRule]:
    when = ["user is owner"] if owner_scoped else None
    return [
        AccessRule(
            permit=Scope(actions=[Action.READ], resource="sql_record::openai_conversations::*"),
            when=when,
        ),
        AccessRule(permit=Scope(actions=[Action.READ], resource=resource), when=when),
    ]


@patch("ogx.core.storage.sqlstore.authorized_sqlstore.get_authenticated_user")
async def test_items_policy_glob_still_matches_after_migration(mock_user, upgrade_backend):
    """A policy whose resource glob names sql_record::conversation_items::* keeps authorizing
    reads after the in-place migration, because the live table keeps its name."""
    conversation_id = "conv_" + "f" * 48
    await _write_v1_database(upgrade_backend, conversation_id)
    mock_user.return_value = User("alice", {"roles": ["admin"]})
    service = await _make_service(upgrade_backend, policy=_items_policy("sql_record::conversation_items::*"))
    listed = await service.list_items(ListItemsRequest(conversation_id=conversation_id, order="asc"))
    assert _texts(listed) == [("msg_0", "v1 0"), ("msg_1", "v1 1")]
    await service.sql_store.sql_store.shutdown()


@patch("ogx.core.storage.sqlstore.authorized_sqlstore.get_authenticated_user")
async def test_policy_glob_on_v1_name_does_not_match_live_table(mock_user, upgrade_backend):
    """A policy that names only the retained v1 table does not authorize reads of the migrated
    items, confirming reads target the live conversation_items table, not conversation_items_v1."""
    conversation_id = "conv_" + "0" * 48
    await _write_v1_database(upgrade_backend, conversation_id)
    mock_user.return_value = User("alice", {"roles": ["admin"]})
    service = await _make_service(upgrade_backend, policy=_items_policy("sql_record::conversation_items_v1::*"))
    listed = await service.list_items(ListItemsRequest(conversation_id=conversation_id, order="asc"))
    assert listed.data == []
    await service.sql_store.sql_store.shutdown()


@patch("ogx.core.storage.sqlstore.authorized_sqlstore.get_authenticated_user")
async def test_owner_isolation_policy_still_matches_after_migration(mock_user, upgrade_backend):
    """The documented owner-isolation policy for conversation_items still authorizes the owner's
    reads after migration. The 'user is owner' condition matches only because the backfill
    preserved ownership, so this also pins that the copied rows are still owned by alice."""
    conversation_id = "conv_" + "1" * 48
    await _write_v1_database(upgrade_backend, conversation_id, owner_principal="alice")
    mock_user.return_value = User("alice", {"roles": ["admin"]})
    service = await _make_service(
        upgrade_backend, policy=_items_policy("sql_record::conversation_items::*", owner_scoped=True)
    )
    listed = await service.list_items(ListItemsRequest(conversation_id=conversation_id, order="asc"))
    assert _texts(listed) == [("msg_0", "v1 0"), ("msg_1", "v1 1")]
    await service.sql_store.sql_store.shutdown()


# --- Primary-key comparison robustness ----------------------------------------


async def test_reversed_composite_key_is_recognized_without_rename(upgrade_backend):
    """A composite-keyed table whose primary key columns the database reports in the opposite
    order is still recognized as migrated and not renamed, because the check compares as a set."""
    store = SqlAlchemySqlStoreImpl(upgrade_backend)
    # Same columns as the composite table, but the two key columns declared in the reverse order.
    await store.create_table(
        ITEMS_TABLE,
        {
            "id": ColumnDefinition(type=ColumnType.STRING, primary_key=True),
            "conversation_id": ColumnDefinition(type=ColumnType.STRING, primary_key=True),
            "created_at": ColumnType.INTEGER,
            "sort_order": ColumnType.INTEGER,
            "item_data": ColumnType.JSON,
            "owner_principal": ColumnType.STRING,
            "access_attributes": ColumnType.JSON,
            "tenant_id": ColumnType.STRING,
        },
    )
    await store.shutdown()

    with _recorded_statements() as statements:
        service = await _make_service(upgrade_backend)

    assert not any("RENAME" in s for s in statements)
    store = service.sql_store.sql_store
    assert set(await store.primary_key_columns(ITEMS_TABLE)) == set(ITEM_KEY_COLUMNS)
    assert not await store.table_exists(ITEMS_V1_TABLE)
    await store.shutdown()


async def test_stray_v1_table_fails_boot_instead_of_skipping_migration(upgrade_backend):
    """A conversation_items_v1 that is not a sibling's work, while conversation_items is still
    id-keyed, is a name collision: the boot fails rather than succeeding with the id-keyed table
    live and the backfill flagged as done."""
    conversation_id = "conv_" + "3" * 48
    await _write_v1_database(upgrade_backend, conversation_id)
    stray = await _v1_store(upgrade_backend, ITEMS_V1_TABLE)
    assert await stray.table_exists(ITEMS_V1_TABLE)  # the store creates tables on first use
    await stray.shutdown()

    with pytest.raises(DBAPIError):
        await _make_service(upgrade_backend)

    store = SqlAlchemySqlStoreImpl(upgrade_backend)
    assert await store.primary_key_columns(ITEMS_TABLE) == ["id"]
    assert not await store.table_exists(MIGRATIONS_TABLE)
    await store.shutdown()


async def test_concurrent_workers_migrate_the_same_database(upgrade_backend):
    """Two workers booting the same not-yet-migrated database at once both finish: the id-keyed
    table is renamed once, its rows copied exactly once with ownership intact, and neither worker
    fails its boot (the rename loses to its sibling and settles, the copy and flag skip conflicts).
    """
    conversation_id = "conv_" + "2" * 48
    await _write_v1_database(upgrade_backend, conversation_id)

    first, second = await asyncio.gather(_make_service(upgrade_backend), _make_service(upgrade_backend))

    store = first.sql_store.sql_store
    assert await store.primary_key_columns(ITEMS_TABLE) == ITEM_KEY_COLUMNS
    assert await store.primary_key_columns(ITEMS_V1_TABLE) == ["id"]
    assert await store.fetch_one(MIGRATIONS_TABLE, where={"name": ITEMS_BACKFILL_KEY}) is not None
    rows = await _raw_items(first, ITEMS_TABLE, conversation_id)
    assert [(row["id"], row["owner_principal"], row["tenant_id"]) for row in rows] == [
        ("msg_0", "alice", "tenant-a"),
        ("msg_1", "alice", "tenant-a"),
    ]
    await store.shutdown()
    await second.sql_store.sql_store.shutdown()
