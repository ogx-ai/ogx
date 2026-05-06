# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from datetime import datetime

from pymongo import AsyncMongoClient
from pymongo.asynchronous.collection import AsyncCollection

from ogx.core.storage.kvstore import KVStore
from ogx.log import get_logger

from ..config import MongoDBKVStoreConfig

log = get_logger(name=__name__, category="providers::utils")


class MongoDBKVStoreImpl(KVStore):
    """MongoDB-backed key-value store implementation."""

    def __init__(self, config: MongoDBKVStoreConfig):
        self.config = config
        self.conn: AsyncMongoClient | None = None

    @property
    def collection(self) -> AsyncCollection:
        if self.conn is None:
            raise RuntimeError("MongoDB connection is not initialized")
        return self.conn[self.config.db][self.config.collection_name]

    async def initialize(self) -> None:
        try:
            # Pass parameters explicitly to satisfy mypy - AsyncMongoClient doesn't accept **dict
            self.conn = AsyncMongoClient(
                host=self.config.host if self.config.host is not None else None,
                port=self.config.port if self.config.port is not None else None,
                username=self.config.user if self.config.user is not None else None,
                password=self.config.password if self.config.password is not None else None,
            )
        except Exception as e:
            log.exception("Could not connect to MongoDB database server")
            raise RuntimeError("Could not connect to MongoDB database server") from e

    def _namespaced_key(self, key: str) -> str:
        if not self.config.namespace:
            return key
        return f"{self.config.namespace}:{key}"

    async def set(self, key: str, value: str, expiration: datetime | None = None) -> None:
        key = self._namespaced_key(key)
        update_query = {"$set": {"value": value, "expiration": expiration}}
        await self.collection.update_one({"key": key}, update_query, upsert=True)

    async def get(self, key: str) -> str | None:
        key = self._namespaced_key(key)
        query = {"key": key}
        result = await self.collection.find_one(query, {"value": 1, "_id": 0})
        return result["value"] if result else None

    async def delete(self, key: str) -> None:
        key = self._namespaced_key(key)
        await self.collection.delete_one({"key": key})

    async def values_in_range(self, start_key: str, end_key: str) -> list[str]:
        start_key = self._namespaced_key(start_key)
        end_key = self._namespaced_key(end_key)
        query = {
            "key": {"$gte": start_key, "$lt": end_key},
        }
        cursor = self.collection.find(query, {"value": 1, "_id": 0}).sort("key", 1)
        result = []
        async for doc in cursor:
            result.append(doc["value"])
        return result

    async def keys_in_range(self, start_key: str, end_key: str) -> list[str]:
        start_key = self._namespaced_key(start_key)
        end_key = self._namespaced_key(end_key)
        query = {"key": {"$gte": start_key, "$lt": end_key}}
        cursor = self.collection.find(query, {"key": 1, "_id": 0}).sort("key", 1)
        # AsyncCursor requires async iteration
        result = []
        async for doc in cursor:
            result.append(doc["key"])
        return result

    async def shutdown(self) -> None:
        if self.conn:
            await self.conn.close()
            self.conn = None
