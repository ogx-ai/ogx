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
)

KVStoreConfig = Annotated[
    RedisKVStoreConfig | SqliteKVStoreConfig | PostgresKVStoreConfig | MongoDBKVStoreConfig, Field(discriminator="type")
]
