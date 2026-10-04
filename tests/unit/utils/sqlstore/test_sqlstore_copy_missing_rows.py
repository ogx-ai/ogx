# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Unit tests for SqlAlchemySqlStoreImpl.copy_missing_rows."""

import pytest

from ogx.core.storage.datatypes import SqliteSqlStoreConfig
from ogx.core.storage.sqlstore.sqlalchemy_sqlstore import SqlAlchemySqlStoreImpl
from ogx_api.internal.sqlstore import ColumnDefinition, ColumnType

KEY_COLUMNS = ["group_id", "id"]


@pytest.fixture
async def store(tmp_path):
    impl = SqlAlchemySqlStoreImpl(SqliteSqlStoreConfig(db_path=str(tmp_path / "copy_missing_rows.db")))
    await impl.create_table(
        "source",
        {
            "id": ColumnDefinition(type=ColumnType.STRING, primary_key=True),
            "group_id": ColumnType.STRING,
            "payload": ColumnType.JSON,
            "source_only": ColumnType.STRING,
        },
    )
    await impl.create_table(
        "target",
        {
            "group_id": ColumnDefinition(type=ColumnType.STRING, primary_key=True),
            "id": ColumnDefinition(type=ColumnType.STRING, primary_key=True),
            "payload": ColumnType.JSON,
            "target_only": ColumnType.INTEGER,
        },
    )
    yield impl
    await impl.shutdown()


def _row(item_id: str, group_id: str = "g1") -> dict:
    return {"id": item_id, "group_id": group_id, "payload": {"text": item_id}, "source_only": "x"}


async def _target_rows(store: SqlAlchemySqlStoreImpl) -> list[dict]:
    result = await store.fetch_all("target", order_by=[("id", "asc")])
    return result.data


async def test_copies_rows_missing_from_target(store):
    await store.insert("source", [_row("a"), _row("b")])

    copied = await store.copy_missing_rows("source", "target", KEY_COLUMNS)

    assert copied == 2
    rows = await _target_rows(store)
    assert [(row["group_id"], row["id"], row["payload"]) for row in rows] == [
        ("g1", "a", {"text": "a"}),
        ("g1", "b", {"text": "b"}),
    ]
    assert all("source_only" not in row and row["target_only"] is None for row in rows)


async def test_rerun_copies_only_new_rows(store):
    await store.insert("source", [_row("a")])
    assert await store.copy_missing_rows("source", "target", KEY_COLUMNS) == 1

    assert await store.copy_missing_rows("source", "target", KEY_COLUMNS) == 0

    await store.insert("source", [_row("b")])
    assert await store.copy_missing_rows("source", "target", KEY_COLUMNS) == 1
    assert [row["id"] for row in await _target_rows(store)] == ["a", "b"]


async def test_existing_target_rows_are_left_alone(store):
    await store.insert("target", {"group_id": "g1", "id": "a", "payload": {"text": "target version"}})
    await store.insert("source", [_row("a"), _row("b")])

    copied = await store.copy_missing_rows("source", "target", KEY_COLUMNS)

    assert copied == 1
    rows = await _target_rows(store)
    assert [(row["id"], row["payload"]) for row in rows] == [
        ("a", {"text": "target version"}),
        ("b", {"text": "b"}),
    ]


async def test_missing_source_table_copies_nothing(store):
    assert await store.copy_missing_rows("no_such_table", "target", KEY_COLUMNS) == 0
    assert await _target_rows(store) == []


async def test_key_column_absent_from_a_table_is_an_error(store):
    with pytest.raises(ValueError, match="Failed to copy rows from source to target"):
        await store.copy_missing_rows("source", "target", ["source_only"])
