# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Annotated

from pydantic import Field

from ogx.core.storage.datatypes import (
    MongoDBKVStoreConfig,
    PostgresKVStoreConfig,
    RedisKVStoreConfig,
    SqliteKVStoreConfig,
    StorageBackendType,
)

KVStoreConfig = Annotated[
    RedisKVStoreConfig | SqliteKVStoreConfig | PostgresKVStoreConfig | MongoDBKVStoreConfig, Field(discriminator="type")
]


def get_pip_packages(store_config: dict | KVStoreConfig) -> list[str]:
    """Get pip packages for KV store config, handling both dict and object cases."""
    if isinstance(store_config, dict):
        store_type = store_config.get("type")
        if store_type == StorageBackendType.KV_SQLITE.value:
            return SqliteKVStoreConfig.pip_packages()
        elif store_type == StorageBackendType.KV_POSTGRES.value:
            return PostgresKVStoreConfig.pip_packages()
        elif store_type == StorageBackendType.KV_REDIS.value:
            return RedisKVStoreConfig.pip_packages()
        elif store_type == StorageBackendType.KV_MONGODB.value:
            return MongoDBKVStoreConfig.pip_packages()
        else:
            raise ValueError(f"Unknown KV store type: {store_type}")
    else:
        return store_config.pip_packages()
