# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Unit tests for the SqlAlchemySqlStoreImpl primitives a table migration is built from:
table_exists, primary_key_columns, rename_table and copy_missing_rows.

They run on SQLite always and on PostgreSQL when ENABLE_POSTGRES_TESTS is set, since
rename_table issues raw DDL and copy_missing_rows reports the driver's rowcount.
"""

import os
import uuid
from dataclasses import dataclass

import pytest
from sqlalchemy import text

from ogx.core.storage.datatypes import PostgresSqlStoreConfig, SqliteSqlStoreConfig
from ogx.core.storage.sqlstore.sqlalchemy_sqlstore import SqlAlchemySqlStoreImpl
from ogx_api.internal.sqlstore import ColumnDefinition, ColumnType

KEY_COLUMNS = ["group_id", "id"]

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


@dataclass
class CopyTables:
    store: SqlAlchemySqlStoreImpl
    source: str
    target: str


@pytest.fixture(params=BACKENDS)
async def tables(request, tmp_path):
    if request.param == "sqlite":
        config = SqliteSqlStoreConfig(db_path=str(tmp_path / "copy_missing_rows.db"))
    else:
        config = PostgresSqlStoreConfig(
            host=os.environ.get("POSTGRES_HOST", "localhost"),
            port=int(os.environ.get("POSTGRES_PORT", "5432")),
            db=os.environ.get("POSTGRES_DB", "ogx"),
            user=os.environ.get("POSTGRES_USER", "ogx"),
            password=os.environ.get("POSTGRES_PASSWORD", "ogx"),
        )
    # Postgres keeps tables between tests, so each test gets its own pair.
    suffix = uuid.uuid4().hex[:8]
    fixture = CopyTables(SqlAlchemySqlStoreImpl(config), f"copy_source_{suffix}", f"copy_target_{suffix}")
    await fixture.store.create_table(
        fixture.source,
        {
            "id": ColumnDefinition(type=ColumnType.STRING, primary_key=True),
            "group_id": ColumnType.STRING,
            "payload": ColumnType.JSON,
            "source_only": ColumnType.STRING,
        },
    )
    await fixture.store.create_table(
        fixture.target,
        {
            "group_id": ColumnDefinition(type=ColumnType.STRING, primary_key=True),
            "id": ColumnDefinition(type=ColumnType.STRING, primary_key=True),
            "payload": ColumnType.JSON,
            "target_only": ColumnType.INTEGER,
        },
    )
    yield fixture
    engine = fixture.store.create_engine()
    async with engine.begin() as conn:
        for table in (fixture.source, fixture.target):
            await conn.execute(text(f'DROP TABLE IF EXISTS "{table}"'))
    await engine.dispose()
    await fixture.store.shutdown()


def _row(item_id: str, group_id: str = "g1") -> dict:
    return {"id": item_id, "group_id": group_id, "payload": {"text": item_id}, "source_only": "x"}


async def _target_rows(tables: CopyTables) -> list[dict]:
    result = await tables.store.fetch_all(tables.target, order_by=[("id", "asc")])
    return result.data


async def test_copies_rows_missing_from_target(tables):
    await tables.store.insert(tables.source, [_row("a"), _row("b")])

    copied = await tables.store.copy_missing_rows(tables.source, tables.target, KEY_COLUMNS)

    assert copied == 2
    rows = await _target_rows(tables)
    assert [(row["group_id"], row["id"], row["payload"]) for row in rows] == [
        ("g1", "a", {"text": "a"}),
        ("g1", "b", {"text": "b"}),
    ]
    assert all("source_only" not in row and row["target_only"] is None for row in rows)


async def test_rerun_copies_only_new_rows(tables):
    await tables.store.insert(tables.source, [_row("a")])
    assert await tables.store.copy_missing_rows(tables.source, tables.target, KEY_COLUMNS) == 1

    assert await tables.store.copy_missing_rows(tables.source, tables.target, KEY_COLUMNS) == 0

    await tables.store.insert(tables.source, [_row("b")])
    assert await tables.store.copy_missing_rows(tables.source, tables.target, KEY_COLUMNS) == 1
    assert [row["id"] for row in await _target_rows(tables)] == ["a", "b"]


async def test_existing_target_rows_are_left_alone(tables):
    await tables.store.insert(tables.target, {"group_id": "g1", "id": "a", "payload": {"text": "target version"}})
    await tables.store.insert(tables.source, [_row("a"), _row("b")])

    copied = await tables.store.copy_missing_rows(tables.source, tables.target, KEY_COLUMNS)

    assert copied == 1
    rows = await _target_rows(tables)
    assert [(row["id"], row["payload"]) for row in rows] == [
        ("a", {"text": "target version"}),
        ("b", {"text": "b"}),
    ]


async def test_missing_source_table_copies_nothing(tables):
    assert await tables.store.copy_missing_rows("no_such_table", tables.target, KEY_COLUMNS) == 0
    assert await _target_rows(tables) == []


async def test_key_column_absent_from_a_table_is_an_error(tables):
    with pytest.raises(ValueError, match=f"Failed to copy rows from {tables.source} to {tables.target}"):
        await tables.store.copy_missing_rows(tables.source, tables.target, ["source_only"])


async def test_table_exists_sees_registered_and_foreign_tables(tables):
    assert await tables.store.table_exists(tables.source)
    assert not await tables.store.table_exists("no_such_table")

    foreign = f"foreign_{uuid.uuid4().hex[:8]}"
    other = SqlAlchemySqlStoreImpl(tables.store.config)
    await other.create_table(foreign, {"id": ColumnDefinition(type=ColumnType.STRING, primary_key=True)})
    await other.insert(foreign, {"id": "x"})
    try:
        assert await tables.store.table_exists(foreign)
    finally:
        engine = other.create_engine()
        async with engine.begin() as conn:
            await conn.execute(text(f'DROP TABLE IF EXISTS "{foreign}"'))
        await engine.dispose()
        await other.shutdown()


async def test_primary_key_columns_follow_the_database(tables):
    assert await tables.store.primary_key_columns(tables.source) == ["id"]
    assert await tables.store.primary_key_columns(tables.target) == KEY_COLUMNS
    assert await tables.store.primary_key_columns("no_such_table") is None


async def test_rename_table_frees_the_old_name_for_a_new_shape(tables):
    """After the rename the rows live under the new name and the old name can be registered again
    with a different primary key, which is how an in-place table migration starts."""
    await tables.store.insert(tables.source, [_row("a"), _row("b")])
    renamed = f"{tables.source}_v1"

    await tables.store.rename_table(tables.source, renamed)

    try:
        assert not await tables.store.table_exists(tables.source)
        assert await tables.store.primary_key_columns(renamed) == ["id"]
        await tables.store.create_table(
            tables.source,
            {
                "group_id": ColumnDefinition(type=ColumnType.STRING, primary_key=True),
                "id": ColumnDefinition(type=ColumnType.STRING, primary_key=True),
                "payload": ColumnType.JSON,
            },
        )
        assert await tables.store.primary_key_columns(tables.source) == KEY_COLUMNS
        assert await tables.store.copy_missing_rows(renamed, tables.source, KEY_COLUMNS) == 2
        assert [row["id"] for row in (await tables.store.fetch_all(tables.source, order_by=[("id", "asc")])).data] == [
            "a",
            "b",
        ]
    finally:
        engine = tables.store.create_engine()
        async with engine.begin() as conn:
            await conn.execute(text(f'DROP TABLE IF EXISTS "{renamed}"'))
        await engine.dispose()


async def test_rename_table_onto_an_existing_name_changes_nothing(tables):
    """The rename runs in a transaction: when it fails, both tables are exactly as before."""
    await tables.store.insert(tables.source, [_row("a")])

    with pytest.raises(Exception, match="already"):
        await tables.store.rename_table(tables.source, tables.target)

    assert await tables.store.primary_key_columns(tables.source) == ["id"]
    assert await tables.store.primary_key_columns(tables.target) == KEY_COLUMNS
    assert [row["id"] for row in (await tables.store.fetch_all(tables.source)).data] == ["a"]
