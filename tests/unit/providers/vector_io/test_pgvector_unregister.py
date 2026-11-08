# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from unittest.mock import AsyncMock, MagicMock

from ogx.providers.remote.vector_io.pgvector.config import PGVectorVectorIOConfig
from ogx.providers.remote.vector_io.pgvector.pgvector import PGVectorVectorIOAdapter


async def test_unregister_drops_the_table_of_a_store_this_replica_never_loaded():
    """Another replica may have created the store, so this one has no index for it in its cache."""
    adapter = PGVectorVectorIOAdapter(
        PGVectorVectorIOConfig(host="localhost", port=5432, db="test_db", user="test_user", password="test_password"),
        inference_api=MagicMock(),
    )
    conn = AsyncMock()
    acquired = AsyncMock()
    acquired.__aenter__ = AsyncMock(return_value=conn)
    acquired.__aexit__ = AsyncMock(return_value=False)
    pool = MagicMock()
    pool.acquire = MagicMock(return_value=acquired)
    adapter._ensure_pool = AsyncMock(return_value=pool)
    adapter.kvstore = AsyncMock()

    await adapter.unregister_vector_store("vs_1234-abcd")

    executed = [call.args[0] for call in conn.execute.await_args_list]
    assert 'DROP TABLE IF EXISTS "vs_vs_1234_abcd"' in executed
