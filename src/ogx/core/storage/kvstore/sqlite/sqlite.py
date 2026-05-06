# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import os
from datetime import datetime

import aiosqlite

from ogx.log import get_logger
from ogx_api.internal.kvstore import KVStore

from ..config import SqliteKVStoreConfig  # type: ignore[attr-defined]

logger = get_logger(name=__name__, category="providers::utils")


class SqliteKVStoreImpl(KVStore):
    """SQLite-backed key-value store implementation."""

    def __init__(self, config: SqliteKVStoreConfig) -> None:
        self.db_path = config.db_path
        self.table_name = "kvstore"
        self._conn: aiosqlite.Connection | None = None

    def __str__(self) -> str:
        return f"SqliteKVStoreImpl(db_path={self.db_path}, table_name={self.table_name})"

    def _is_memory_db(self) -> bool:
        """Check if this is an in-memory database."""
        return self.db_path == ":memory:" or "mode=memory" in self.db_path

    async def initialize(self) -> None:
        # Skip directory creation for in-memory databases and file: URIs
        if not self._is_memory_db() and not self.db_path.startswith("file:"):
            db_dir = os.path.dirname(self.db_path)
            if db_dir:  # Only create if there's a directory component
                os.makedirs(db_dir, exist_ok=True)

        # Only use persistent connection for in-memory databases
        # File-based databases use connection-per-operation to avoid hangs
        if self._is_memory_db():
            self._conn = await aiosqlite.connect(self.db_path)
            await self._conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self.table_name} (
                    key TEXT PRIMARY KEY,
                    value TEXT,
                    expiration TIMESTAMP
                )
            """
            )
            await self._conn.commit()
        else:
            # For file-based databases, just create the table
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {self.table_name} (
                        key TEXT PRIMARY KEY,
                        value TEXT,
                        expiration TIMESTAMP
                    )
                """
                )
                await db.commit()

    async def shutdown(self) -> None:
        """Close the persistent connection (only for in-memory databases)."""
        if self._conn:
            await self._conn.close()
            self._conn = None

    async def set(self, key: str, value: str, expiration: datetime | None = None) -> None:
        if self._conn:
            # In-memory database with persistent connection
            await self._conn.execute(
                f"INSERT OR REPLACE INTO {self.table_name} (key, value, expiration) VALUES (?, ?, ?)",
                (key, value, expiration),
            )
            await self._conn.commit()
        else:
            # File-based database with connection per operation
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute(
                    f"INSERT OR REPLACE INTO {self.table_name} (key, value, expiration) VALUES (?, ?, ?)",
                    (key, value, expiration),
                )
                await db.commit()

    async def get(self, key: str) -> str | None:
        if self._conn:
            # In-memory database with persistent connection
            async with self._conn.execute(
                f"SELECT value, expiration FROM {self.table_name} WHERE key = ?", (key,)
            ) as cursor:
                row = await cursor.fetchone()
                if row is None:
                    return None
                value, expiration = row
                if not isinstance(value, str):
                    logger.warning("Expected string value for key, returning None", key=key, value_type=type(value))
                    return None
                return value
        else:
            # File-based database with connection per operation
            async with aiosqlite.connect(self.db_path) as db:
                async with db.execute(
                    f"SELECT value, expiration FROM {self.table_name} WHERE key = ?", (key,)
                ) as cursor:
                    row = await cursor.fetchone()
                    if row is None:
                        return None
                    value, expiration = row
                    if not isinstance(value, str):
                        logger.warning("Expected string value for key, returning None", key=key, value_type=type(value))
                        return None
                    return value

    async def delete(self, key: str) -> None:
        if self._conn:
            # In-memory database with persistent connection
            await self._conn.execute(f"DELETE FROM {self.table_name} WHERE key = ?", (key,))
            await self._conn.commit()
        else:
            # File-based database with connection per operation
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute(f"DELETE FROM {self.table_name} WHERE key = ?", (key,))
                await db.commit()

    async def values_in_range(self, start_key: str, end_key: str) -> list[str]:
        if self._conn:
            # In-memory database with persistent connection
            async with self._conn.execute(
                f"SELECT key, value, expiration FROM {self.table_name} WHERE key >= ? AND key <= ?",
                (start_key, end_key),
            ) as cursor:
                result = []
                async for row in cursor:
                    _, value, _ = row
                    result.append(value)
                return result
        else:
            # File-based database with connection per operation
            async with aiosqlite.connect(self.db_path) as db:
                async with db.execute(
                    f"SELECT key, value, expiration FROM {self.table_name} WHERE key >= ? AND key <= ?",
                    (start_key, end_key),
                ) as cursor:
                    result = []
                    async for row in cursor:
                        _, value, _ = row
                        result.append(value)
                    return result

    async def keys_in_range(self, start_key: str, end_key: str) -> list[str]:
        """Get all keys in the given range."""
        if self._conn:
            # In-memory database with persistent connection
            cursor = await self._conn.execute(
                f"SELECT key FROM {self.table_name} WHERE key >= ? AND key <= ?",
                (start_key, end_key),
            )
            rows = await cursor.fetchall()
            return [row[0] for row in rows]
        else:
            # File-based database with connection per operation
            async with aiosqlite.connect(self.db_path) as db:
                cursor = await db.execute(
                    f"SELECT key FROM {self.table_name} WHERE key >= ? AND key <= ?",
                    (start_key, end_key),
                )
                rows = await cursor.fetchall()
                return [row[0] for row in rows]
