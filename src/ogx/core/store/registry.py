# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import asyncio
import time
from contextlib import asynccontextmanager
from typing import Protocol

import pydantic

from ogx.core.datatypes import RoutableObjectWithProvider
from ogx.core.storage.datatypes import KVStoreReference
from ogx.core.storage.kvstore import KVStore, kvstore_impl
from ogx.log import get_logger

logger = get_logger(__name__, category="core::registry")


class DistributionRegistry(Protocol):
    """Protocol for distribution registries that store and retrieve routable objects."""

    async def get_all(self) -> list[RoutableObjectWithProvider]: ...

    async def initialize(self) -> None: ...

    async def get(self, type: str, identifier: str) -> RoutableObjectWithProvider | None: ...

    def get_cached(self, type: str, identifier: str) -> RoutableObjectWithProvider | None: ...

    async def update(self, obj: RoutableObjectWithProvider) -> RoutableObjectWithProvider: ...

    async def register(self, obj: RoutableObjectWithProvider) -> bool: ...

    async def delete(self, type: str, identifier: str) -> None: ...


REGISTER_PREFIX = "distributions:registry"
KEY_VERSION = "v10"
KEY_FORMAT = f"{REGISTER_PREFIX}:{KEY_VERSION}::" + "{type}:{identifier}"


def _get_registry_key_range() -> tuple[str, str]:
    """Returns the start and end keys for the registry range query."""
    start_key = f"{REGISTER_PREFIX}:{KEY_VERSION}"
    return start_key, f"{start_key}\xff"


def _parse_registry_values(values: list[str]) -> list[RoutableObjectWithProvider]:
    """Utility function to parse registry values into RoutableObjectWithProvider objects."""
    all_objects = []
    for value in values:
        try:
            obj = pydantic.TypeAdapter(RoutableObjectWithProvider).validate_json(value)
            all_objects.append(obj)
        except pydantic.ValidationError as e:
            logger.error("Error parsing registry value", raw_value=value, error=str(e))
            continue

    return all_objects


class DiskDistributionRegistry(DistributionRegistry):
    """KVStore-backed distribution registry that persists objects to disk."""

    def __init__(self, kvstore: KVStore):
        self.kvstore = kvstore

    async def initialize(self) -> None:
        pass

    def get_cached(self, type: str, identifier: str) -> RoutableObjectWithProvider | None:
        # Disk registry does not have a cache
        raise NotImplementedError("Disk registry does not have a cache")

    async def get_all(self) -> list[RoutableObjectWithProvider]:
        start_key, end_key = _get_registry_key_range()
        values = await self.kvstore.values_in_range(start_key, end_key)
        return _parse_registry_values(values)

    async def get(self, type: str, identifier: str) -> RoutableObjectWithProvider | None:
        json_str = await self.kvstore.get(KEY_FORMAT.format(type=type, identifier=identifier))
        if not json_str:
            return None

        try:
            return pydantic.TypeAdapter(RoutableObjectWithProvider).validate_json(json_str)
        except pydantic.ValidationError as e:
            logger.error(
                "Error parsing registry value",
                resource_type=type,
                identifier=identifier,
                raw_value=json_str,
                error=str(e),
            )
            return None

    async def update(self, obj: RoutableObjectWithProvider) -> RoutableObjectWithProvider:
        await self.kvstore.set(
            KEY_FORMAT.format(type=obj.type, identifier=obj.identifier),
            obj.model_dump_json(),
        )
        return obj

    async def register(self, obj: RoutableObjectWithProvider) -> bool:
        existing_obj = await self.get(obj.type, obj.identifier)
        if existing_obj:
            if existing_obj == obj:
                return True
            # Allow re-registration when the incoming object is a subset of the
            # existing one (every explicitly-set field matches).  This covers
            # server restarts where the config-provided object lacks mutable
            # fields (e.g. ``owner``) that were added during initial registration.
            # Genuinely conflicting field values still raise an error.
            incoming_data = obj.model_dump()
            existing_data = existing_obj.model_dump()
            conflicts = {
                field: (incoming_data[field], existing_data[field])
                for field in obj.model_fields_set
                if incoming_data[field] != existing_data[field]
            }
            if conflicts:
                raise ValueError(
                    f"Object of type '{obj.type}' and identifier '{obj.identifier}' already exists "
                    f"with conflicting field values: {conflicts}. "
                    "Unregister it first if you want to replace it."
                )
            logger.debug("Re-registration is a no-op (subset match)", obj_type=obj.type, identifier=obj.identifier)
            return True

        await self.kvstore.set(
            KEY_FORMAT.format(type=obj.type, identifier=obj.identifier),
            obj.model_dump_json(),
        )
        return True

    async def delete(self, type: str, identifier: str) -> None:
        await self.kvstore.delete(KEY_FORMAT.format(type=type, identifier=identifier))


class CachedDiskDistributionRegistry(DiskDistributionRegistry):
    """Distribution registry with an in-memory cache layer over the disk-backed KVStore."""

    def __init__(self, kvstore: KVStore, cache_ttl_seconds: float = 5.0):
        super().__init__(kvstore)
        self.cache: dict[tuple[str, str], RoutableObjectWithProvider] = {}
        self._initialized = False
        self._initialize_lock = asyncio.Lock()
        self._cache_lock = asyncio.Lock()
        self._cache_ttl_seconds = cache_ttl_seconds
        self._last_refresh_time = 0.0

    @asynccontextmanager
    async def _locked_cache(self):
        """Context manager for safely accessing the cache with a lock."""
        async with self._cache_lock:
            yield self.cache

    async def _ensure_initialized(self):
        """Ensures the registry is initialized before operations."""
        if self._initialized:
            return

        async with self._initialize_lock:
            if self._initialized:
                return

            start_key, end_key = _get_registry_key_range()
            values = await self.kvstore.values_in_range(start_key, end_key)
            objects = _parse_registry_values(values)

            async with self._locked_cache() as cache:
                for obj in objects:
                    cache_key = (obj.type, obj.identifier)
                    cache[cache_key] = obj

            self._initialized = True

    async def initialize(self) -> None:
        await self._ensure_initialized()

    def get_cached(self, type: str, identifier: str) -> RoutableObjectWithProvider | None:
        return self.cache.get((type, identifier), None)

    def _should_refresh_cache(self) -> bool:
        """Check if cache should be refreshed based on TTL."""
        current_time = time.time()
        return (current_time - self._last_refresh_time) >= self._cache_ttl_seconds

    async def _refresh_cache_from_db(self) -> None:
        """Refresh cache from database if TTL has expired."""
        if not self._should_refresh_cache():
            return

        start_key, end_key = _get_registry_key_range()
        values = await self.kvstore.values_in_range(start_key, end_key)
        objects = _parse_registry_values(values)

        async with self._locked_cache() as cache:
            for obj in objects:
                cache_key = (obj.type, obj.identifier)
                cache[cache_key] = obj

        self._last_refresh_time = time.time()

    async def get_all(self) -> list[RoutableObjectWithProvider]:
        await self._ensure_initialized()

        # Refresh cache from database to handle multi-worker scenarios
        # This ensures we see objects created by other workers
        # Uses TTL to avoid hammering the database
        await self._refresh_cache_from_db()

        async with self._locked_cache() as cache:
            return list(cache.values())

    async def get(self, type: str, identifier: str) -> RoutableObjectWithProvider | None:
        await self._ensure_initialized()
        cache_key = (type, identifier)

        # First check the cache
        async with self._locked_cache() as cache:
            cached_obj = cache.get(cache_key, None)
            if cached_obj is not None:
                return cached_obj

        # If not in cache, check the database (handles multi-worker scenarios)
        obj = await super().get(type, identifier)
        if obj is not None:
            # Update cache with the newly found object
            async with self._locked_cache() as cache:
                cache[cache_key] = obj

        return obj

    async def register(self, obj: RoutableObjectWithProvider) -> bool:
        await self._ensure_initialized()
        success = await super().register(obj)

        if success:
            cache_key = (obj.type, obj.identifier)
            async with self._locked_cache() as cache:
                cache[cache_key] = obj

        return success

    async def update(self, obj: RoutableObjectWithProvider) -> RoutableObjectWithProvider:
        result = await super().update(obj)
        cache_key = (obj.type, obj.identifier)
        async with self._locked_cache() as cache:
            cache[cache_key] = obj
        return result

    async def delete(self, type: str, identifier: str) -> None:
        await super().delete(type, identifier)
        cache_key = (type, identifier)
        async with self._locked_cache() as cache:
            if cache_key in cache:
                del cache[cache_key]


async def create_dist_registry(
    metadata_store: KVStoreReference, distro_name: str, cache_ttl_seconds: float = 5.0
) -> tuple[CachedDiskDistributionRegistry, KVStore]:
    """Create and initialize a cached distribution registry backed by a KVStore.

    Args:
        metadata_store: KVStore reference for storing registry metadata.
        distro_name: Name of the distribution.
        cache_ttl_seconds: Time-to-live for cache entries in seconds.

    Returns:
        A tuple of (initialized CachedDiskDistributionRegistry, underlying KVStore).
    """
    # instantiate kvstore for storing and retrieving distribution metadata
    dist_kvstore = await kvstore_impl(metadata_store)
    dist_registry = CachedDiskDistributionRegistry(dist_kvstore, cache_ttl_seconds=cache_ttl_seconds)
    await dist_registry.initialize()
    return dist_registry, dist_kvstore
