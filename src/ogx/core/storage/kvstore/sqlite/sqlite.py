# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import os
from datetime import UTC, datetime

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
        self._namespace = config.namespace
        self._conn: aiosqlite.Connection | None = None

    def __str__(self) -> str:
        return f"SqliteKVStoreImpl(db_path={self.db_path}, table_name={self.table_name})"

    def _is_memory_db(self) -> bool:
        """Check if this is an in-memory database."""
        return self.db_path == ":memory:" or "mode=memory" in self.db_path

    def _namespaced_key(self, key: str) -> str:
        if not self._namespace:
            return key
        return f"{self._namespace}:{key}"

    def _strip_namespace(self, key: str) -> str:
        if self._namespace and key.startswith(f"{self._namespace}:"):
            return key[len(self._namespace) + 1 :]
        return key

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
        key = self._namespaced_key(key)
        exp_str = expiration.isoformat() if expiration else None
        if self._conn:
            await self._conn.execute(
                f"INSERT OR REPLACE INTO {self.table_name} (key, value, expiration) VALUES (?, ?, ?)",
                (key, value, exp_str),
            )
            await self._conn.commit()
        else:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute(
                    f"INSERT OR REPLACE INTO {self.table_name} (key, value, expiration) VALUES (?, ?, ?)",
                    (key, value, exp_str),
                )
                await db.commit()

    async def get(self, key: str) -> str | None:
        key = self._namespaced_key(key)
        now = datetime.now(tz=UTC).isoformat()
        query = f"SELECT value FROM {self.table_name} WHERE key = ? AND (expiration IS NULL OR expiration > ?)"
        if self._conn:
            async with self._conn.execute(query, (key, now)) as cursor:
                row = await cursor.fetchone()
                if row is None:
                    return None
                value = row[0]
                if not isinstance(value, str):
                    logger.warning("Expected string value for key, returning None", key=key, value_type=type(value))
                    return None
                return value
        else:
            async with aiosqlite.connect(self.db_path) as db:
                async with db.execute(query, (key, now)) as cursor:
                    row = await cursor.fetchone()
                    if row is None:
                        return None
                    value = row[0]
                    if not isinstance(value, str):
                        logger.warning("Expected string value for key, returning None", key=key, value_type=type(value))
                        return None
                    return value

    async def delete(self, key: str) -> None:
        key = self._namespaced_key(key)
        if self._conn:
            await self._conn.execute(f"DELETE FROM {self.table_name} WHERE key = ?", (key,))
            await self._conn.commit()
        else:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute(f"DELETE FROM {self.table_name} WHERE key = ?", (key,))
                await db.commit()

    async def values_in_range(self, start_key: str, end_key: str) -> list[str]:
        start_key = self._namespaced_key(start_key)
        end_key = self._namespaced_key(end_key)
        now = datetime.now(tz=UTC).isoformat()
        query = (
            f"SELECT value FROM {self.table_name} "
            f"WHERE key >= ? AND key < ? AND (expiration IS NULL OR expiration > ?) "
            f"ORDER BY key"
        )
        if self._conn:
            async with self._conn.execute(query, (start_key, end_key, now)) as cursor:
                return [row[0] async for row in cursor]
        else:
            async with aiosqlite.connect(self.db_path) as db:
                async with db.execute(query, (start_key, end_key, now)) as cursor:
                    return [row[0] async for row in cursor]

    async def keys_in_range(self, start_key: str, end_key: str) -> list[str]:
        start_key = self._namespaced_key(start_key)
        end_key = self._namespaced_key(end_key)
        now = datetime.now(tz=UTC).isoformat()
        query = (
            f"SELECT key FROM {self.table_name} "
            f"WHERE key >= ? AND key < ? AND (expiration IS NULL OR expiration > ?) "
            f"ORDER BY key"
        )
        if self._conn:
            cursor = await self._conn.execute(query, (start_key, end_key, now))
            rows = await cursor.fetchall()
            return [self._strip_namespace(row[0]) for row in rows]
        else:
            async with aiosqlite.connect(self.db_path) as db:
                cursor = await db.execute(query, (start_key, end_key, now))
                rows = await cursor.fetchall()
                return [self._strip_namespace(row[0]) for row in rows]
