# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
from collections.abc import Mapping, Sequence
from typing import Any, Literal, cast

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Float,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    event,
    inspect,
    select,
    text,
)
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.ext.asyncio.engine import AsyncEngine
from sqlalchemy.ext.asyncio.session import AsyncSession
from sqlalchemy.sql.elements import ColumnElement

from ogx.core.storage.datatypes import PostgresSqlStoreConfig, SqlAlchemySqlStoreConfig
from ogx.log import get_logger
from ogx_api import PaginatedResponse
from ogx_api.internal.sqlstore import ColumnDefinition, ColumnType, SqlStore

logger = get_logger(name=__name__, category="providers::utils")

TYPE_MAPPING: dict[ColumnType, Any] = {
    ColumnType.INTEGER: Integer,
    ColumnType.STRING: String,
    ColumnType.FLOAT: Float,
    ColumnType.BOOLEAN: Boolean,
    ColumnType.DATETIME: DateTime,
    ColumnType.TEXT: Text,
    ColumnType.JSON: JSON,
}


def _build_where_expr(column: ColumnElement[Any], value: Any) -> ColumnElement[Any]:
    """Return a SQLAlchemy expression for a where condition.

    `value` may be a simple scalar (equality) or a mapping like {">": 123}.
    The returned expression is a SQLAlchemy ColumnElement usable in query.where(...).
    """
    if isinstance(value, Mapping):
        if len(value) != 1:
            raise ValueError(f"Operator mapping must have a single operator, got: {value}")
        op, operand = next(iter(value.items()))
        if op == "==" or op == "=":
            return cast(ColumnElement[Any], column == operand)
        if op == ">":
            return cast(ColumnElement[Any], column > operand)
        if op == "<":
            return cast(ColumnElement[Any], column < operand)
        if op == ">=":
            return cast(ColumnElement[Any], column >= operand)
        if op == "<=":
            return cast(ColumnElement[Any], column <= operand)
        raise ValueError(f"Unsupported operator '{op}' in where mapping")
    return cast(ColumnElement[Any], column == value)


class SqlAlchemySqlStoreImpl(SqlStore):
    """SQLAlchemy-based SQL store implementation supporting SQLite and PostgreSQL backends."""

    def __init__(self, config: SqlAlchemySqlStoreConfig) -> None:
        self.config = config
        self._is_sqlite_backend = "sqlite" in self.config.engine_str
        self._engine: AsyncEngine | None = None  # Lazy initialization
        self.async_session: async_sessionmaker[AsyncSession] | None = None
        self.metadata = MetaData()
        self._pending_columns: dict[
            str, list[tuple[str, ColumnType, bool]]
        ] = {}  # table -> [(col_name, col_type, nullable)]

    async def _ensure_engine(self) -> None:
        """Lazy initialization: create engine on first use in the current event loop.

        This fixes event loop mismatch issues when Stack is initialized in a different
        event loop (e.g., ThreadPoolExecutor) than request handling (uvicorn's loop).
        """
        if self._engine is None:
            # Create engine in the current running event loop
            self._engine = self.create_engine()
            self.async_session = async_sessionmaker(self._engine)

            # Create all tables that were registered during initialization
            if self.metadata.tables:
                async with self._engine.begin() as conn:
                    await conn.run_sync(self.metadata.create_all, checkfirst=True)

            # Add all pending columns that were queued during initialization
            for table_name, columns in self._pending_columns.items():
                for col_name, col_type, nullable in columns:
                    await self._add_column_now(table_name, col_name, col_type, nullable)
            self._pending_columns.clear()

    async def shutdown(self) -> None:
        """Dispose of the async engine and close all connections."""
        if self._engine:
            await self._engine.dispose()
            self._engine = None

    def create_engine(self) -> AsyncEngine:
        # Configure connection args for better concurrency support
        connect_args = {}
        engine_kwargs: dict[str, Any] = {"pool_pre_ping": self.config.pool_pre_ping}
        if self._is_sqlite_backend:
            # SQLite-specific optimizations for concurrent access
            # With WAL mode, most locks resolve in milliseconds, but allow up to 5s for edge cases
            connect_args["timeout"] = 5.0
            connect_args["check_same_thread"] = False  # Allow usage across asyncio tasks
        elif isinstance(self.config, PostgresSqlStoreConfig):
            engine_kwargs["pool_size"] = self.config.pool_size
            engine_kwargs["max_overflow"] = self.config.max_overflow
            if self.config.pool_recycle >= 0:
                engine_kwargs["pool_recycle"] = self.config.pool_recycle

        engine = create_async_engine(
            self.config.engine_str,
            connect_args=connect_args,
            **engine_kwargs,
        )

        # Enable WAL mode for SQLite to support concurrent readers and writers
        if self._is_sqlite_backend:

            @event.listens_for(engine.sync_engine, "connect")
            def set_sqlite_pragma(dbapi_conn: Any, connection_record: Any) -> None:
                cursor = dbapi_conn.cursor()
                # Enable Write-Ahead Logging for better concurrency
                cursor.execute("PRAGMA journal_mode=WAL")
                # Set busy timeout to 5 seconds (retry instead of immediate failure)
                # With WAL mode, locks should be brief; if we hit 5s there's a bigger issue
                cursor.execute("PRAGMA busy_timeout=5000")
                # Use NORMAL synchronous mode for better performance (still safe with WAL)
                cursor.execute("PRAGMA synchronous=NORMAL")
                cursor.close()

        return engine

    async def create_table(
        self,
        table: str,
        schema: Mapping[str, ColumnType | ColumnDefinition],
    ) -> None:
        # Don't create engine yet - just store table metadata
        # Engine will be created on first data operation
        if not schema:
            raise ValueError(f"No columns defined for table '{table}'.")

        sqlalchemy_columns: list[Column[Any]] = []

        for col_name, col_props in schema.items():
            col_type = None
            is_primary_key = False
            is_nullable = True

            if isinstance(col_props, ColumnType):
                col_type = col_props
            elif isinstance(col_props, ColumnDefinition):
                col_type = col_props.type
                is_primary_key = col_props.primary_key
                is_nullable = col_props.nullable

            sqlalchemy_type = TYPE_MAPPING.get(col_type)
            if not sqlalchemy_type:
                raise ValueError(f"Unsupported column type '{col_type}' for column '{col_name}'.")

            sqlalchemy_columns.append(
                Column(col_name, sqlalchemy_type, primary_key=is_primary_key, nullable=is_nullable)
            )

        # Register table in metadata - actual creation happens in _ensure_engine()
        if table not in self.metadata.tables:
            Table(table, self.metadata, *sqlalchemy_columns)
        # If table already exists in metadata, we're done (no need to recreate)

    async def insert(self, table: str, data: Mapping[str, Any] | Sequence[Mapping[str, Any]]) -> None:
        await self._ensure_engine()  # Lazy init in current event loop
        assert self.async_session is not None  # _ensure_engine guarantees this
        async with self.async_session() as session:
            await session.execute(self.metadata.tables[table].insert(), data)
            await session.commit()

    async def upsert(
        self,
        table: str,
        data: Mapping[str, Any],
        conflict_columns: list[str],
        update_columns: list[str] | None = None,
    ) -> None:
        await self._ensure_engine()  # Lazy init in current event loop
        assert self.async_session is not None  # _ensure_engine guarantees this
        table_obj = self.metadata.tables[table]
        dialect_insert = self._get_dialect_insert(table_obj)
        insert_stmt = dialect_insert.values(**data)

        if update_columns is None:
            update_columns = [col for col in data.keys() if col not in conflict_columns]

        update_mapping = {col: getattr(insert_stmt.excluded, col) for col in update_columns}
        conflict_cols = [table_obj.c[col] for col in conflict_columns]

        stmt = insert_stmt.on_conflict_do_update(index_elements=conflict_cols, set_=update_mapping)

        async with self.async_session() as session:
            await session.execute(stmt)
            await session.commit()

    async def fetch_all(
        self,
        table: str,
        where: Mapping[str, Any] | None = None,
        where_sql: str | None = None,
        where_sql_params: Mapping[str, Any] | None = None,
        limit: int | None = None,
        order_by: list[tuple[str, Literal["asc", "desc"]]] | None = None,
        cursor: tuple[str, str] | None = None,
    ) -> PaginatedResponse:
        await self._ensure_engine()  # Lazy init in current event loop
        assert self.async_session is not None  # _ensure_engine guarantees this
        async with self.async_session() as session:
            table_obj = self.metadata.tables[table]
            query = select(table_obj)

            if where:
                for key, value in where.items():
                    query = query.where(_build_where_expr(table_obj.c[key], value))

            if where_sql:
                clause = text(where_sql)
                if where_sql_params:
                    clause = clause.bindparams(**where_sql_params)
                query = query.where(clause)

            # Handle cursor-based pagination
            if cursor:
                # Validate cursor tuple format
                if not isinstance(cursor, tuple) or len(cursor) != 2:
                    raise ValueError(f"Cursor must be a tuple of (key_column, cursor_id), got: {cursor}")

                # Require order_by for cursor pagination
                if not order_by:
                    raise ValueError("order_by is required when using cursor pagination")

                # Only support single-column ordering for cursor pagination
                if len(order_by) != 1:
                    raise ValueError(
                        f"Cursor pagination only supports single-column ordering, got {len(order_by)} columns"
                    )

                cursor_key_column, cursor_id = cursor
                order_column, order_direction = order_by[0]

                # Verify cursor_key_column exists
                if cursor_key_column not in table_obj.c:
                    raise ValueError(f"Cursor key column '{cursor_key_column}' not found in table '{table}'")

                # Get cursor value for the order column
                cursor_query = select(table_obj.c[order_column]).where(table_obj.c[cursor_key_column] == cursor_id)
                cursor_result = await session.execute(cursor_query)
                cursor_row = cursor_result.fetchone()

                if not cursor_row:
                    raise ValueError(f"Record with {cursor_key_column}='{cursor_id}' not found in table '{table}'")

                cursor_value = cursor_row[0]

                # Apply cursor condition based on sort direction
                if order_direction == "desc":
                    query = query.where(table_obj.c[order_column] < cursor_value)
                else:
                    query = query.where(table_obj.c[order_column] > cursor_value)

            # Apply ordering
            if order_by:
                if not isinstance(order_by, list):
                    raise ValueError(
                        f"order_by must be a list of tuples (column, order={['asc', 'desc']}), got {order_by}"
                    )
                for order in order_by:
                    if not isinstance(order, tuple):
                        raise ValueError(
                            f"order_by must be a list of tuples (column, order={['asc', 'desc']}), got {order_by}"
                        )
                    name, order_type = order
                    if name not in table_obj.c:
                        raise ValueError(f"Column '{name}' not found in table '{table}'")
                    if order_type == "asc":
                        query = query.order_by(table_obj.c[name].asc())
                    elif order_type == "desc":
                        query = query.order_by(table_obj.c[name].desc())
                    else:
                        raise ValueError(f"Invalid order '{order_type}' for column '{name}'")

            # Fetch limit + 1 to determine has_more
            fetch_limit = limit
            if limit:
                fetch_limit = limit + 1

            if fetch_limit:
                query = query.limit(fetch_limit)

            result = await session.execute(query)
            # Iterate directly - if no rows, list comprehension yields empty list
            rows = [dict(row._mapping) for row in result]

            # Always return pagination result
            has_more = False
            if limit and len(rows) > limit:
                has_more = True
                rows = rows[:limit]

            return PaginatedResponse(data=rows, has_more=has_more)

    async def fetch_one(
        self,
        table: str,
        where: Mapping[str, Any] | None = None,
        where_sql: str | None = None,
        where_sql_params: Mapping[str, Any] | None = None,
        order_by: list[tuple[str, Literal["asc", "desc"]]] | None = None,
    ) -> dict[str, Any] | None:
        result = await self.fetch_all(table, where, where_sql, where_sql_params, limit=1, order_by=order_by)
        if not result.data:
            return None
        return result.data[0]

    async def update(
        self,
        table: str,
        data: Mapping[str, Any],
        where: Mapping[str, Any],
        where_sql: str | None = None,
        where_sql_params: Mapping[str, Any] | None = None,
    ) -> None:
        await self._ensure_engine()  # Lazy init in current event loop
        assert self.async_session is not None  # _ensure_engine guarantees this
        if not where:
            raise ValueError("where is required for update")

        async with self.async_session() as session:
            stmt = self.metadata.tables[table].update()
            for key, value in where.items():
                stmt = stmt.where(_build_where_expr(self.metadata.tables[table].c[key], value))
            if where_sql:
                clause = text(where_sql)
                if where_sql_params:
                    clause = clause.bindparams(**where_sql_params)
                stmt = stmt.where(clause)
            await session.execute(stmt, data)
            await session.commit()

    async def delete(
        self,
        table: str,
        where: Mapping[str, Any],
        where_sql: str | None = None,
        where_sql_params: Mapping[str, Any] | None = None,
    ) -> None:
        await self._ensure_engine()  # Lazy init in current event loop
        assert self.async_session is not None  # _ensure_engine guarantees this
        if not where:
            raise ValueError("where is required for delete")

        async with self.async_session() as session:
            stmt = self.metadata.tables[table].delete()
            for key, value in where.items():
                stmt = stmt.where(_build_where_expr(self.metadata.tables[table].c[key], value))
            if where_sql:
                clause = text(where_sql)
                if where_sql_params:
                    clause = clause.bindparams(**where_sql_params)
                stmt = stmt.where(clause)
            await session.execute(stmt)
            await session.commit()

    async def add_column_if_not_exists(
        self,
        table: str,
        column_name: str,
        column_type: ColumnType,
        nullable: bool = True,
    ) -> None:
        """Queue a column to be added when engine is created, or add it now if engine exists."""
        if self._engine is None:
            # Engine not created yet - queue this column addition for later
            if table not in self._pending_columns:
                self._pending_columns[table] = []
            self._pending_columns[table].append((column_name, column_type, nullable))
        else:
            # Engine already exists - add column immediately
            await self._add_column_now(table, column_name, column_type, nullable)

    async def _add_column_now(
        self,
        table: str,
        column_name: str,
        column_type: ColumnType,
        nullable: bool = True,
    ) -> None:
        """Actually add a column to an existing table if the column doesn't already exist."""
        assert self._engine is not None  # Only called when engine exists
        try:
            async with self._engine.begin() as conn:

                def check_column_exists(sync_conn: Any) -> tuple[bool, bool]:
                    inspector = inspect(sync_conn)

                    table_names = inspector.get_table_names()
                    if table not in table_names:
                        return False, False  # table doesn't exist, column doesn't exist

                    existing_columns = inspector.get_columns(table)
                    column_names = [col["name"] for col in existing_columns]

                    return True, column_name in column_names  # table exists, column exists or not

                table_exists, column_exists = await conn.run_sync(check_column_exists)
                if not table_exists or column_exists:
                    return

                sqlalchemy_type = TYPE_MAPPING.get(column_type)
                if not sqlalchemy_type:
                    raise ValueError(f"Unsupported column type '{column_type}' for column '{column_name}'.")

                # Create the ALTER TABLE statement
                # Note: We need to get the dialect-specific type name
                dialect = self._engine.dialect
                type_impl = sqlalchemy_type()
                compiled_type = type_impl.compile(dialect=dialect)

                nullable_clause = "" if nullable else " NOT NULL"
                add_column_sql = text(f"ALTER TABLE {table} ADD COLUMN {column_name} {compiled_type}{nullable_clause}")

                await conn.execute(add_column_sql)
        except Exception as e:
            # If any error occurs during migration, log it but don't fail
            # The table creation will handle adding the column
            logger.error("Error adding column to table", column_name=column_name, table=table, error=str(e))
            pass

    def _get_dialect_insert(self, table: Table) -> Any:
        if self._is_sqlite_backend:
            return sqlite_insert(table)
        else:
            return pg_insert(table)
