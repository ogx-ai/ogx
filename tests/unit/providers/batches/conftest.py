# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Shared fixtures for batches provider unit tests."""

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from ogx.core.storage.datatypes import SqliteSqlStoreConfig, SqlStoreReference
from ogx.core.storage.sqlstore.authorized_sqlstore import AuthorizedSqlStore
from ogx.core.storage.sqlstore.sqlstore import _sqlstore_impl, register_sqlstore_backends
from ogx.providers.inline.batches.reference.batches import ReferenceBatchesImpl
from ogx.providers.inline.batches.reference.config import ReferenceBatchesImplConfig


@pytest.fixture
async def provider():
    """Create a test provider instance with temporary database."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_batches.db"
        backend_name = "sql_batches_test"
        register_sqlstore_backends({backend_name: SqliteSqlStoreConfig(db_path=str(db_path))})
        config = ReferenceBatchesImplConfig(sqlstore=SqlStoreReference(backend=backend_name, table_name="batches"))

        # Create sql_store and mock APIs
        base_sql_store = _sqlstore_impl(config.sqlstore)
        sql_store = AuthorizedSqlStore(base_sql_store, policy=[])
        mock_inference = AsyncMock()
        mock_files = AsyncMock()
        mock_models = AsyncMock()

        provider = ReferenceBatchesImpl(config, mock_inference, mock_files, mock_models, sql_store)
        await provider.initialize()

        # unit tests should not require background processing
        provider.process_batches = False

        yield provider

        await provider.shutdown()


@pytest.fixture
def sample_batch_data():
    """Sample batch data for testing."""
    return {
        "input_file_id": "file_abc123",
        "endpoint": "/v1/chat/completions",
        "completion_window": "24h",
        "metadata": {"test": "true", "priority": "high"},
    }
