# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import pytest

from ogx.core.jobs.queue import JobQueue
from ogx.core.storage.datatypes import SqliteSqlStoreConfig
from ogx.core.storage.sqlstore.sqlalchemy_sqlstore import SqlAlchemySqlStoreImpl


@pytest.fixture
async def queue(tmp_path):
    # A file-based DB (not :memory:) so concurrent connections share one database,
    # matching how the server and worker processes share the queue in production.
    store = SqlAlchemySqlStoreImpl(SqliteSqlStoreConfig(db_path=str(tmp_path / "jobs.db")))
    q = JobQueue(store, table_name="jobs", lease_ttl_seconds=60)
    await q.initialize()
    yield q
    await store.shutdown()
