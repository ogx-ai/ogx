# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

# Copyright (c) The OGX Contributors.
# All rights reserved.
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from __future__ import annotations

import asyncio
from collections import defaultdict
from datetime import datetime
from typing import cast

from ogx.core.storage.datatypes import (
    KVStoreReference,
    MongoDBKVStoreConfig,
    PostgresKVStoreConfig,
    RedisKVStoreConfig,
    SqliteKVStoreConfig,
    StorageBackendConfig,
)
from ogx_api.internal.kvstore import KVStore

from .config import KVStoreConfig


def kvstore_dependencies() -> list[str]:
    """
    Returns all possible kvstore dependencies for registry/provider specifications.

    NOTE: For specific kvstore implementations, use config.pip_packages instead.
    This function returns the union of all dependencies for cases where the specific
    kvstore type is not known at declaration time (e.g., provider registries).
    """
    return ["aiosqlite", "psycopg2-binary", "redis", "pymongo"]


class InmemoryKVStoreImpl(KVStore):
    """In-memory key-value store implementation for testing and ephemeral usage."""

    def __init__(self) -> None:
        self._store: dict[str, str] = {}

    async def initialize(self) -> None:
        pass

    async def get(self, key: str) -> str | None:
        return self._store.get(key)

    async def set(self, key: str, value: str, expiration: datetime | None = None) -> None:
        self._store[key] = value

    async def values_in_range(self, start_key: str, end_key: str) -> list[str]:
        return [self._store[key] for key in self._store.keys() if key >= start_key and key < end_key]

    async def keys_in_range(self, start_key: str, end_key: str) -> list[str]:
        """Get all keys in the given range."""
        return [key for key in self._store.keys() if key >= start_key and key < end_key]

    async def delete(self, key: str) -> None:
        self._store.pop(key, None)

    async def shutdown(self) -> None:
        self._store.clear()


_KVSTORE_BACKENDS: dict[str, KVStoreConfig] = {}
_KVSTORE_INSTANCES: dict[tuple[str, str], KVStore] = {}
_KVSTORE_LOCKS: defaultdict[tuple[str, str], asyncio.Lock] = defaultdict(asyncio.Lock)


def register_kvstore_backends(backends: dict[str, StorageBackendConfig]) -> None:
    """Register the set of available KV store backends for reference resolution."""
    global _KVSTORE_BACKENDS
    global _KVSTORE_INSTANCES
    global _KVSTORE_LOCKS

    _KVSTORE_BACKENDS.clear()
    _KVSTORE_INSTANCES.clear()
    _KVSTORE_LOCKS.clear()
    for name, cfg in backends.items():
        typed_cfg = cast(KVStoreConfig, cfg)
        _KVSTORE_BACKENDS[name] = typed_cfg


async def kvstore_impl(reference: KVStoreReference) -> KVStore:
    """Get or create a KVStore instance for the given store reference.

    Args:
        reference: A reference specifying the backend name and namespace.

    Returns:
        An initialized KVStore instance for the referenced backend.

    Raises:
        ValueError: If the backend name is unknown or the backend type is unsupported.
    """
    backend_name = reference.backend
    cache_key = (backend_name, reference.namespace)

    existing = _KVSTORE_INSTANCES.get(cache_key)
    if existing:
        return existing

    backend_config = _KVSTORE_BACKENDS.get(backend_name)
    if backend_config is None:
        raise ValueError(f"Unknown KVStore backend '{backend_name}'. Registered backends: {sorted(_KVSTORE_BACKENDS)}")

    lock = _KVSTORE_LOCKS[cache_key]
    async with lock:
        existing = _KVSTORE_INSTANCES.get(cache_key)
        if existing:
            return existing

        config = backend_config.model_copy()
        config.namespace = reference.namespace

        impl: KVStore
        if isinstance(config, RedisKVStoreConfig):
            from .redis import RedisKVStoreImpl  # type: ignore[attr-defined]

            impl = RedisKVStoreImpl(config)
        elif isinstance(config, SqliteKVStoreConfig):
            from .sqlite import SqliteKVStoreImpl  # type: ignore[attr-defined]

            impl = SqliteKVStoreImpl(config)
        elif isinstance(config, PostgresKVStoreConfig):
            from .postgres import PostgresKVStoreImpl  # type: ignore[attr-defined]

            impl = PostgresKVStoreImpl(config)
        elif isinstance(config, MongoDBKVStoreConfig):
            from .mongodb import MongoDBKVStoreImpl

            impl = MongoDBKVStoreImpl(config)
        else:
            raise ValueError(f"Unknown kvstore type {config.type}")

        await impl.initialize()
        _KVSTORE_INSTANCES[cache_key] = impl
        return impl


async def shutdown_kvstore_backends() -> None:
    """Shutdown all cached KV store instances."""
    global _KVSTORE_INSTANCES
    for instance in _KVSTORE_INSTANCES.values():
        await instance.shutdown()
    _KVSTORE_INSTANCES.clear()
