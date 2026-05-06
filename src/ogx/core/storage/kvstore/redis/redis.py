# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from datetime import datetime

from redis.asyncio import Redis  # type: ignore[import-not-found]

from ogx_api.internal.kvstore import KVStore

from ..config import RedisKVStoreConfig


class RedisKVStoreImpl(KVStore):
    """Redis-backed key-value store implementation."""

    def __init__(self, config: RedisKVStoreConfig):
        self.config = config
        self._redis: Redis | None = None

    async def initialize(self) -> None:
        self._redis = Redis.from_url(self.config.url)

    def _client(self) -> Redis:
        if self._redis is None:
            raise RuntimeError("Redis client not initialized")
        return self._redis

    def _namespaced_key(self, key: str) -> str:
        if not self.config.namespace:
            return key
        return f"{self.config.namespace}:{key}"

    async def set(self, key: str, value: str, expiration: datetime | None = None) -> None:
        key = self._namespaced_key(key)
        client = self._client()
        await client.set(key, value)
        if expiration:
            await client.expireat(key, expiration)

    async def get(self, key: str) -> str | None:
        key = self._namespaced_key(key)
        client = self._client()
        value = await client.get(key)
        if value is None:
            return None
        await client.ttl(key)
        if isinstance(value, bytes):
            return value.decode("utf-8")
        if isinstance(value, str):
            return value
        return str(value)

    async def delete(self, key: str) -> None:
        key = self._namespaced_key(key)
        await self._client().delete(key)

    async def values_in_range(self, start_key: str, end_key: str) -> list[str]:
        start_key = self._namespaced_key(start_key)
        end_key = self._namespaced_key(end_key)
        client = self._client()
        cursor = 0
        pattern = start_key + "*"  # Match all keys starting with start_key prefix
        matching_keys: list[str | bytes] = []
        while True:
            cursor, keys = await client.scan(cursor, match=pattern, count=1000)

            for key in keys:
                key_str = key.decode("utf-8") if isinstance(key, bytes) else key
                if start_key <= key_str <= end_key:
                    matching_keys.append(key)

            if cursor == 0:
                break

        # Then fetch all values in a single MGET call
        if matching_keys:
            values = await client.mget(matching_keys)
            return [
                value.decode("utf-8") if isinstance(value, bytes) else value for value in values if value is not None
            ]

        return []

    async def keys_in_range(self, start_key: str, end_key: str) -> list[str]:
        """Get all keys in the given range."""
        start_key = self._namespaced_key(start_key)
        end_key = self._namespaced_key(end_key)
        client = self._client()
        cursor = 0
        pattern = start_key + "*"
        result: list[str] = []
        while True:
            cursor, keys = await client.scan(cursor, match=pattern, count=1000)
            for key in keys:
                key_str = key.decode("utf-8") if isinstance(key, bytes) else str(key)
                if start_key <= key_str <= end_key:
                    result.append(key_str)
            if cursor == 0:
                break
        return result

    async def shutdown(self) -> None:
        if self._redis:
            await self._redis.close()
            self._redis = None
