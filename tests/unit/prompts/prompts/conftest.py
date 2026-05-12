# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import random

import pytest

from ogx.core.prompts.prompts import PromptServiceConfig, PromptServiceImpl
from ogx.core.storage.datatypes import (
    InferenceStoreReference,
    KVStoreReference,
    ServerStoresConfig,
    SqliteKVStoreConfig,
    SqliteSqlStoreConfig,
    SqlStoreReference,
    StorageConfig,
)
from ogx.core.storage.kvstore import register_kvstore_backends
from ogx.core.storage.sqlstore.sqlstore import register_sqlstore_backends


@pytest.fixture
async def temp_prompt_store(tmp_path_factory):
    unique_id = f"prompt_store_{random.randint(1, 1000000)}"
    temp_dir = tmp_path_factory.getbasetemp()
    db_path = str(temp_dir / f"{unique_id}.db")
    sql_db_path = str(temp_dir / f"{unique_id}_sql.db")

    from ogx.core.datatypes import StackConfig

    storage = StorageConfig(
        backends={
            "kv_test": SqliteKVStoreConfig(db_path=db_path),
            "sql_test": SqliteSqlStoreConfig(db_path=sql_db_path),
        },
        stores=ServerStoresConfig(
            metadata=KVStoreReference(backend="kv_test", namespace="registry"),
            inference=InferenceStoreReference(backend="sql_test", table_name="inference"),
            conversations=SqlStoreReference(backend="sql_test", table_name="conversations"),
            prompts=SqlStoreReference(backend="sql_test", table_name="prompts"),
            connectors=SqlStoreReference(backend="sql_test", table_name="connectors"),
        ),
    )

    # Backends must be registered before PromptServiceImpl constructor calls authorized_sqlstore()
    register_kvstore_backends({"kv_test": storage.backends["kv_test"]})
    register_sqlstore_backends({"sql_test": storage.backends["sql_test"]})

    mock_run_config = StackConfig(
        distro_name="test-distribution",
        apis=[],
        providers={},
        storage=storage,
    )
    config = PromptServiceConfig(config=mock_run_config)
    store = PromptServiceImpl(config, deps={})
    await store.initialize()

    yield store
