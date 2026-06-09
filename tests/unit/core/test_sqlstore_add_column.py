# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Unit tests for SqlAlchemySqlStoreImpl._add_column_now error handling."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ogx.core.storage.datatypes import SqliteSqlStoreConfig
from ogx.core.storage.sqlstore.sqlalchemy_sqlstore import SqlAlchemySqlStoreImpl
from ogx_api.internal.sqlstore import ColumnType


@pytest.fixture
async def store():
    config = SqliteSqlStoreConfig(db_path=":memory:")
    impl = SqlAlchemySqlStoreImpl(config)
    await impl.create_table("test_table", {"id": ColumnType.STRING})
    await impl._ensure_engine()
    yield impl
    await impl.shutdown()


def _make_mock_engine(side_effect: Exception) -> MagicMock:
    """Build a mock engine whose begin() context manager raises on execute()."""
    mock_engine = MagicMock()
    mock_engine.dialect = MagicMock()
    mock_conn = AsyncMock()
    mock_conn.run_sync = AsyncMock(return_value=(True, False))
    mock_conn.execute = AsyncMock(side_effect=side_effect)
    mock_ctx = AsyncMock()
    mock_ctx.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_ctx.__aexit__ = AsyncMock(return_value=False)
    mock_engine.begin = MagicMock(return_value=mock_ctx)
    return mock_engine


def _with_mock_engine(store: SqlAlchemySqlStoreImpl, error: Exception):
    """Context manager that temporarily replaces the engine with a mock, restoring it on exit."""
    import contextlib

    @contextlib.contextmanager
    def _ctx():
        real_engine = store._engine
        store._engine = _make_mock_engine(error)
        try:
            yield
        finally:
            store._engine = real_engine

    return _ctx()


class TestAddColumnNow:
    async def test_add_new_column_succeeds(self, store: SqlAlchemySqlStoreImpl):
        await store._add_column_now("test_table", "new_col", ColumnType.STRING, nullable=True)
        async with store._engine.begin() as conn:
            from sqlalchemy import inspect as sa_inspect

            def get_cols(sync_conn):
                inspector = sa_inspect(sync_conn)
                return [c["name"] for c in inspector.get_columns("test_table")]

            columns = await conn.run_sync(get_cols)
        assert "new_col" in columns

    async def test_already_exists_column_is_handled_gracefully(self, store: SqlAlchemySqlStoreImpl):
        await store._add_column_now("test_table", "dup_col", ColumnType.STRING, nullable=True)
        await store._add_column_now("test_table", "dup_col", ColumnType.STRING, nullable=True)

    async def test_duplicate_column_error_does_not_propagate(self, store: SqlAlchemySqlStoreImpl):
        with _with_mock_engine(store, Exception("duplicate column name: dup_col")):
            await store._add_column_now("test_table", "dup_col", ColumnType.STRING)

    async def test_already_exists_error_does_not_propagate(self, store: SqlAlchemySqlStoreImpl):
        with _with_mock_engine(store, Exception("column dup_col already exists")):
            await store._add_column_now("test_table", "dup_col", ColumnType.STRING)

    async def test_permission_error_propagates(self, store: SqlAlchemySqlStoreImpl):
        with _with_mock_engine(store, Exception("permission denied for table test_table")):
            with pytest.raises(RuntimeError, match="Failed to add column bad_col to test_table"):
                await store._add_column_now("test_table", "bad_col", ColumnType.STRING)

    async def test_connection_error_propagates(self, store: SqlAlchemySqlStoreImpl):
        with _with_mock_engine(store, Exception("could not connect to server")):
            with pytest.raises(RuntimeError, match="Failed to add column conn_col to test_table"):
                await store._add_column_now("test_table", "conn_col", ColumnType.STRING)

    async def test_propagated_error_chains_original(self, store: SqlAlchemySqlStoreImpl):
        original = Exception("some db error")
        with _with_mock_engine(store, original):
            with pytest.raises(RuntimeError) as exc_info:
                await store._add_column_now("test_table", "x", ColumnType.STRING)
            assert exc_info.value.__cause__ is original

    async def test_already_exists_logged_at_debug(self, store: SqlAlchemySqlStoreImpl):
        with _with_mock_engine(store, Exception("column col already exists")):
            with patch("ogx.core.storage.sqlstore.sqlalchemy_sqlstore.logger") as mock_logger:
                await store._add_column_now("test_table", "col", ColumnType.STRING)
                mock_logger.debug.assert_called_once()

    async def test_table_not_exists_returns_early(self, store: SqlAlchemySqlStoreImpl):
        await store._add_column_now("nonexistent_table", "col", ColumnType.STRING, nullable=True)
