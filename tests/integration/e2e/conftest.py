# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import asyncio
import os

import asyncpg
import pytest

# Import fixtures from common module to make them available in this test directory
from tests.integration.fixtures.common import (  # noqa: F401
    openai_client,
    require_server,
)


def pytest_configure(config):
    os.environ["OGX_TEST_LOG_STDERR"] = "0"


class SyncDB:
    """Synchronous wrapper around an asyncpg connection.

    Uses a single event loop for both the connection and all queries
    to avoid asyncpg's "attached to a different loop" error.
    """

    def __init__(self):
        self._loop = asyncio.new_event_loop()
        self._conn = self._loop.run_until_complete(
            asyncpg.connect(
                host=os.environ.get("POSTGRES_HOST", "127.0.0.1"),
                port=int(os.environ.get("POSTGRES_PORT", "5432")),
                database=os.environ.get("POSTGRES_DB", "ogx"),
                user=os.environ.get("POSTGRES_USER", "ogx"),
                password=os.environ.get("POSTGRES_PASSWORD", "ogx"),
            )
        )

    def fetchrow(self, query, *args):
        return self._loop.run_until_complete(self._conn.fetchrow(query, *args))

    def fetch(self, query, *args):
        return self._loop.run_until_complete(self._conn.fetch(query, *args))

    def close(self):
        self._loop.run_until_complete(self._conn.close())
        self._loop.close()


@pytest.fixture(scope="session")
def db():
    """Direct PostgreSQL connection for verifying data persistence."""
    sync_db = SyncDB()
    yield sync_db
    sync_db.close()
