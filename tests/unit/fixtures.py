# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import pytest

from ogx.core.storage.kvstore.config import SqliteKVStoreConfig
from ogx.core.storage.kvstore.sqlite import SqliteKVStoreImpl
from ogx.core.store.registry import CachedDiskDistributionRegistry, DiskDistributionRegistry


@pytest.fixture(scope="function")
async def sqlite_kvstore(tmp_path):
    db_path = tmp_path / "test_kv.db"
    kvstore_config = SqliteKVStoreConfig(db_path=db_path.as_posix())
    kvstore = SqliteKVStoreImpl(kvstore_config)
    await kvstore.initialize()
    yield kvstore


@pytest.fixture(scope="function")
async def disk_dist_registry(sqlite_kvstore):
    registry = DiskDistributionRegistry(sqlite_kvstore)
    await registry.initialize()
    yield registry


@pytest.fixture(scope="function")
async def cached_disk_dist_registry(sqlite_kvstore):
    # Use cache_ttl_seconds=0 for tests to ensure immediate synchronization
    registry = CachedDiskDistributionRegistry(sqlite_kvstore, cache_ttl_seconds=0)
    await registry.initialize()
    yield registry
