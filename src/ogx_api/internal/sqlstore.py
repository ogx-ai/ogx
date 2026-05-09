# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from collections.abc import Mapping, Sequence
from enum import Enum
from typing import Any, Literal, Protocol

from pydantic import BaseModel

from ogx_api import PaginatedResponse


class ColumnType(Enum):
    """Supported SQL column data types for table schema definitions."""

    INTEGER = "INTEGER"
    STRING = "STRING"
    TEXT = "TEXT"
    FLOAT = "FLOAT"
    BOOLEAN = "BOOLEAN"
    JSON = "JSON"
    DATETIME = "DATETIME"


class ColumnDefinition(BaseModel):
    """Schema definition for a single SQL table column."""

    type: ColumnType
    primary_key: bool = False
    nullable: bool = True
    default: Any = None


class SqlStore(Protocol):
    """Protocol for common SQL-store functionality."""

    async def create_table(self, table: str, schema: Mapping[str, ColumnType | ColumnDefinition]) -> None: ...

    async def insert(self, table: str, data: Mapping[str, Any] | Sequence[Mapping[str, Any]]) -> None: ...

    async def upsert(
        self,
        table: str,
        data: Mapping[str, Any],
        conflict_columns: list[str],
        update_columns: list[str] | None = None,
    ) -> None: ...

    async def fetch_all(
        self,
        table: str,
        where: Mapping[str, Any] | None = None,
        where_sql: str | None = None,
        where_sql_params: Mapping[str, Any] | None = None,
        limit: int | None = None,
        order_by: list[tuple[str, Literal["asc", "desc"]]] | None = None,
        cursor: tuple[str, str] | None = None,
    ) -> PaginatedResponse: ...

    async def fetch_one(
        self,
        table: str,
        where: Mapping[str, Any] | None = None,
        where_sql: str | None = None,
        where_sql_params: Mapping[str, Any] | None = None,
        order_by: list[tuple[str, Literal["asc", "desc"]]] | None = None,
    ) -> dict[str, Any] | None: ...

    async def update(
        self,
        table: str,
        data: Mapping[str, Any],
        where: Mapping[str, Any],
        where_sql: str | None = None,
        where_sql_params: Mapping[str, Any] | None = None,
    ) -> None: ...

    async def delete(
        self,
        table: str,
        where: Mapping[str, Any],
        where_sql: str | None = None,
        where_sql_params: Mapping[str, Any] | None = None,
    ) -> None: ...

    async def add_column_if_not_exists(
        self,
        table: str,
        column_name: str,
        column_type: ColumnType,
        nullable: bool = True,
    ) -> None: ...

    async def shutdown(self) -> None: ...


__all__ = ["ColumnDefinition", "ColumnType", "SqlStore"]
