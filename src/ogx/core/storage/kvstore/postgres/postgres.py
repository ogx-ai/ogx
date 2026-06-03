# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import asyncio
from collections.abc import Callable, Coroutine
from datetime import datetime
from typing import TypeVar

import asyncpg  # type: ignore[import-untyped]

from ogx.log import get_logger
from ogx_api.internal.kvstore import KVStore

from ..config import PostgresKVStoreConfig

log = get_logger(name=__name__, category="providers::utils")

T = TypeVar("T")


class PostgresKVStoreImpl(KVStore):
    """PostgreSQL-backed key-value store implementation."""

    def __init__(self, config: PostgresKVStoreConfig):
        self.config = config
        self._pool: asyncpg.Pool | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._table_created = False

    async def initialize(self) -> None:
        pass

    def _build_ssl(self) -> object:
        if self.config.ssl_mode == "verify-full" and self.config.ca_cert_path:
            import ssl as _ssl

            return _ssl.create_default_context(cafile=self.config.ca_cert_path)
        if self.config.ssl_mode and self.config.ssl_mode != "disable":
            return self.config.ssl_mode
        return None

    async def _acquire(self) -> asyncpg.Pool:
        loop = asyncio.get_running_loop()
        if self._pool is not None and self._loop is not loop:
            # Pool was created in a different event loop (e.g., during init in a
            # temporary asyncio.run() loop). Discard it -- the old connections are
            # already dead since that loop is closed.
            self._pool = None
            self._table_created = False

        if self._pool is None:
            try:
                self._pool = await asyncpg.create_pool(
                    host=self.config.host,
                    port=int(self.config.port),
                    database=self.config.db,
                    user=self.config.user,
                    password=self.config.password.get_secret_value() if self.config.password else None,
                    ssl=self._build_ssl(),
                    min_size=self.config.pool_size,
                    max_size=self.config.pool_size + self.config.max_overflow,
                    command_timeout=self.config.command_timeout,
                )
                self._loop = loop
            except Exception as e:
                log.exception("Could not connect to PostgreSQL database server")
                raise RuntimeError("Could not connect to PostgreSQL database server") from e

        if not self._table_created:
            async with self._pool.acquire() as conn:
                await conn.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {self.config.table_name} (
                        key TEXT PRIMARY KEY,
                        value TEXT,
                        expiration TIMESTAMPTZ
                    )
                    """
                )
                await conn.execute(
                    f"""
                    CREATE INDEX IF NOT EXISTS idx_{self.config.table_name}_expiration
                    ON {self.config.table_name} (expiration)
                    WHERE expiration IS NOT NULL
                    """
                )
                self._table_created = True

        return self._pool

    async def _execute_with_retry(self, fn: Callable[[asyncpg.Connection], Coroutine[None, None, T]]) -> T:
        """Execute fn with a pooled connection, retrying once on connection error."""
        pool = await self._acquire()
        try:
            async with pool.acquire() as conn:
                return await fn(conn)
        except (
            asyncpg.exceptions.ConnectionDoesNotExistError,
            asyncpg.exceptions.InterfaceError,
            OSError,
            RuntimeError,
        ):
            log.warning("PostgreSQL connection lost, expiring pool connections")
            await pool.expire_connections()
            async with pool.acquire() as conn:
                return await fn(conn)

    def _namespaced_key(self, key: str) -> str:
        if not self.config.namespace:
            return key
        return f"{self.config.namespace}:{key}"

    def _strip_namespace(self, key: str) -> str:
        if self.config.namespace and key.startswith(f"{self.config.namespace}:"):
            return key[len(self.config.namespace) + 1 :]
        return key

    async def set(self, key: str, value: str, expiration: datetime | None = None) -> None:
        key = self._namespaced_key(key)

        async def _do(conn: asyncpg.Connection) -> None:
            await conn.execute(
                f"""
                INSERT INTO {self.config.table_name} (key, value, expiration)
                VALUES ($1, $2, $3)
                ON CONFLICT (key) DO UPDATE
                SET value = EXCLUDED.value, expiration = EXCLUDED.expiration
                """,
                key,
                value,
                expiration,
            )

        await self._execute_with_retry(_do)

    async def get(self, key: str) -> str | None:
        key = self._namespaced_key(key)

        async def _do(conn: asyncpg.Connection) -> str | None:
            row = await conn.fetchrow(
                f"""
                SELECT value FROM {self.config.table_name}
                WHERE key = $1
                AND (expiration IS NULL OR expiration > NOW())
                """,
                key,
            )
            return row["value"] if row else None

        return await self._execute_with_retry(_do)

    async def delete(self, key: str) -> None:
        key = self._namespaced_key(key)

        async def _do(conn: asyncpg.Connection) -> None:
            await conn.execute(
                f"DELETE FROM {self.config.table_name} WHERE key = $1",
                key,
            )

        await self._execute_with_retry(_do)

    async def values_in_range(self, start_key: str, end_key: str) -> list[str]:
        start_key = self._namespaced_key(start_key)
        end_key = self._namespaced_key(end_key)

        async def _do(conn: asyncpg.Connection) -> list[str]:
            rows = await conn.fetch(
                f"""
                SELECT value FROM {self.config.table_name}
                WHERE key >= $1 AND key < $2
                AND (expiration IS NULL OR expiration > NOW())
                ORDER BY key
                """,
                start_key,
                end_key,
            )
            return [row["value"] for row in rows]

        return await self._execute_with_retry(_do)

    async def keys_in_range(self, start_key: str, end_key: str) -> list[str]:
        start_key = self._namespaced_key(start_key)
        end_key = self._namespaced_key(end_key)

        async def _do(conn: asyncpg.Connection) -> list[str]:
            rows = await conn.fetch(
                f"""
                SELECT key FROM {self.config.table_name}
                WHERE key >= $1 AND key < $2
                AND (expiration IS NULL OR expiration > NOW())
                ORDER BY key
                """,
                start_key,
                end_key,
            )
            return [self._strip_namespace(row["key"]) for row in rows]

        return await self._execute_with_retry(_do)

    async def shutdown(self) -> None:
        if self._pool:
            await self._pool.close()
            self._pool = None
