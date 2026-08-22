# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import os
import subprocess
import time

import pytest

from ogx.core.storage.kvstore.config import PostgresKVStoreConfig
from ogx.core.storage.kvstore.postgres.postgres import PostgresKVStoreImpl

POSTGRES_ENABLED = os.environ.get("ENABLE_POSTGRES_TESTS", "").lower() == "true"

pytestmark = pytest.mark.skipif(
    not POSTGRES_ENABLED,
    reason="PostgreSQL tests disabled (set ENABLE_POSTGRES_TESTS=true)",
)


def get_postgres_config():
    return PostgresKVStoreConfig(
        host=os.environ.get("POSTGRES_HOST", "localhost"),
        port=int(os.environ.get("POSTGRES_PORT", "5432")),
        db=os.environ.get("POSTGRES_DB", "ogx"),
        user=os.environ.get("POSTGRES_USER", "ogx"),
        password=os.environ.get("POSTGRES_PASSWORD", "ogx"),
        table_name="test_kvstore_resilience",
    )


def _find_postgres_container():
    """Find the running postgres container ID."""
    result = subprocess.run(
        ["docker", "ps", "-q", "--filter", "ancestor=postgres:15"],
        capture_output=True,
        text=True,
    )
    container_id = result.stdout.strip()
    if not container_id:
        # Fallback: match any postgres container
        result = subprocess.run(
            ["docker", "ps", "-q", "--filter", "name=postgres"],
            capture_output=True,
            text=True,
        )
        container_id = result.stdout.strip()
    if not container_id:
        pytest.skip("No postgres container found to restart")
    # Take first if multiple
    return container_id.split("\n")[0]


def _restart_postgres(container_id: str, timeout: int = 30):
    """Restart the postgres container and wait for it to accept connections."""
    subprocess.run(["docker", "restart", container_id], check=True)
    deadline = time.time() + timeout
    while time.time() < deadline:
        result = subprocess.run(
            [
                "docker",
                "exec",
                container_id,
                "pg_isready",
                "-U",
                "ogx",
            ],
            capture_output=True,
        )
        if result.returncode == 0:
            return
        time.sleep(1)
    raise TimeoutError("PostgreSQL did not become ready after restart")


async def test_kvstore_recovers_after_postgres_restart():
    config = get_postgres_config()
    store = PostgresKVStoreImpl(config)
    await store.initialize()

    # Baseline: kvstore works
    await store.set("test_key", "before_restart")
    value = await store.get("test_key")
    assert value == "before_restart"

    # Restart postgres
    container_id = _find_postgres_container()
    _restart_postgres(container_id)

    # After restart, kvstore should still work.
    # Without reconnect logic this will raise InterfaceError.
    await store.set("test_key", "after_restart")
    value = await store.get("test_key")
    assert value == "after_restart"

    await store.shutdown()
