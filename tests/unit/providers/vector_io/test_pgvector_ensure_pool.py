# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ogx.providers.remote.vector_io.pgvector.config import PGVectorVectorIOConfig
from ogx.providers.remote.vector_io.pgvector.pgvector import PGVectorVectorIOAdapter


@pytest.fixture
def pgvector_config():
    return PGVectorVectorIOConfig(
        host="localhost",
        port=5432,
        db="test_db",
        user="test_user",
        password="test_password",
    )


@pytest.fixture
def adapter(pgvector_config):
    return PGVectorVectorIOAdapter(pgvector_config, inference_api=MagicMock(), files_api=None)


async def test_ensure_pool_creates_extension_before_pool(adapter):
    """Verify CREATE EXTENSION runs via standalone connection before asyncpg.create_pool().

    Regression test for https://github.com/ogx-ai/ogx/issues/6164 — the pool's
    init callback calls register_vector() which requires the vector type to exist.
    If the extension is created after pool creation, register_vector() crashes with
    ValueError: unknown type: public.vector.
    """
    call_order = []

    mock_conn = AsyncMock()
    mock_conn.fetchval = AsyncMock(return_value=None)
    mock_conn.execute = AsyncMock()
    mock_conn.close = AsyncMock()

    mock_pool = MagicMock()
    mock_pool.close = AsyncMock()
    pool_acm = AsyncMock()
    pool_acm.__aenter__ = AsyncMock(return_value=mock_conn)
    pool_acm.__aexit__ = AsyncMock(return_value=False)
    mock_pool.acquire = MagicMock(return_value=pool_acm)

    async def track_connect(**kwargs):
        call_order.append("connect")
        return mock_conn

    async def track_create_pool(**kwargs):
        call_order.append("create_pool")
        return mock_pool

    async def track_create_extension(conn):
        call_order.append("create_extension")

    with (
        patch("ogx.providers.remote.vector_io.pgvector.pgvector.asyncpg.connect", side_effect=track_connect),
        patch("ogx.providers.remote.vector_io.pgvector.pgvector.asyncpg.create_pool", side_effect=track_create_pool),
        patch(
            "ogx.providers.remote.vector_io.pgvector.pgvector.create_vector_extension",
            side_effect=track_create_extension,
        ),
        patch(
            "ogx.providers.remote.vector_io.pgvector.pgvector.check_extension_version",
            new_callable=AsyncMock,
            return_value=None,
        ),
    ):
        await adapter._ensure_pool()

    assert call_order.index("create_extension") < call_order.index("create_pool"), (
        f"Extension must be created before pool, but got: {call_order}"
    )
    mock_conn.close.assert_called_once()


async def test_ensure_pool_skips_extension_creation_when_exists(adapter):
    """When the vector extension already exists, skip CREATE EXTENSION and proceed to pool creation."""
    mock_conn = AsyncMock()
    mock_conn.execute = AsyncMock()
    mock_conn.close = AsyncMock()

    mock_pool = MagicMock()
    mock_pool.close = AsyncMock()
    pool_acm = AsyncMock()
    pool_acm.__aenter__ = AsyncMock(return_value=mock_conn)
    pool_acm.__aexit__ = AsyncMock(return_value=False)
    mock_pool.acquire = MagicMock(return_value=pool_acm)

    with (
        patch(
            "ogx.providers.remote.vector_io.pgvector.pgvector.asyncpg.connect", new_callable=AsyncMock
        ) as mock_connect,
        patch(
            "ogx.providers.remote.vector_io.pgvector.pgvector.asyncpg.create_pool", new_callable=AsyncMock
        ) as mock_create_pool,
        patch(
            "ogx.providers.remote.vector_io.pgvector.pgvector.create_vector_extension",
            new_callable=AsyncMock,
        ) as mock_create_ext,
        patch(
            "ogx.providers.remote.vector_io.pgvector.pgvector.check_extension_version",
            new_callable=AsyncMock,
            return_value="0.8.0",
        ),
    ):
        mock_connect.return_value = mock_conn
        mock_create_pool.return_value = mock_pool

        await adapter._ensure_pool()

    mock_create_ext.assert_not_called()
    mock_conn.close.assert_called_once()


async def test_ensure_pool_closes_standalone_connection_on_error(adapter):
    """The standalone connection must be closed even if extension creation fails."""
    mock_conn = AsyncMock()
    mock_conn.close = AsyncMock()

    with (
        patch(
            "ogx.providers.remote.vector_io.pgvector.pgvector.asyncpg.connect", new_callable=AsyncMock
        ) as mock_connect,
        patch(
            "ogx.providers.remote.vector_io.pgvector.pgvector.check_extension_version",
            new_callable=AsyncMock,
            return_value=None,
        ),
        patch(
            "ogx.providers.remote.vector_io.pgvector.pgvector.create_vector_extension",
            new_callable=AsyncMock,
            side_effect=RuntimeError("permission denied"),
        ),
    ):
        mock_connect.return_value = mock_conn

        with pytest.raises(RuntimeError, match="permission denied"):
            await adapter._ensure_pool()

    mock_conn.close.assert_called_once()
