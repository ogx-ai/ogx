# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


from ogx.core.storage.kvstore.config import SqliteKVStoreConfig
from ogx.core.storage.kvstore.sqlite.sqlite import SqliteKVStoreImpl


async def test_memory_kvstore_persistence_behavior():
    """Test that :memory: database doesn't persist across instances."""
    config = SqliteKVStoreConfig(db_path=":memory:")

    # First instance
    store1 = SqliteKVStoreImpl(config)
    await store1.initialize()
    await store1.set("persist_test", "should_not_persist")
    await store1.shutdown()

    # Second instance with same config
    store2 = SqliteKVStoreImpl(config)
    await store2.initialize()

    # Data should not be present
    result = await store2.get("persist_test")
    assert result is None

    await store2.shutdown()
