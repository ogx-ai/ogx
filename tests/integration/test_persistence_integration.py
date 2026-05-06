# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import yaml

from ogx.core.datatypes import StackConfig
from ogx.core.storage.datatypes import (
    PostgresKVStoreConfig,
    PostgresSqlStoreConfig,
    SqliteKVStoreConfig,
    SqliteSqlStoreConfig,
)


def test_starter_distribution_config_loads_and_resolves():
    """Integration: Actual starter config should parse and have correct storage structure."""
    with open("ogx/distributions/starter/config.yaml") as f:
        config_dict = yaml.safe_load(f)

    config = StackConfig(**config_dict)

    # Config should have named backends and explicit store references
    assert config.storage is not None
    assert "kv_default" in config.storage.backends
    assert "sql_default" in config.storage.backends
    assert isinstance(config.storage.backends["kv_default"], SqliteKVStoreConfig)
    assert isinstance(config.storage.backends["sql_default"], SqliteSqlStoreConfig)

    stores = config.storage.stores
    assert stores.metadata is not None
    assert stores.metadata.backend == "kv_default"
    assert stores.metadata.namespace == "registry"

    assert stores.inference is not None
    assert stores.inference.backend == "sql_default"
    assert stores.inference.table_name == "inference_store"
    assert stores.inference.max_write_queue_size > 0
    assert stores.inference.num_writers > 0

    assert stores.conversations is not None
    assert stores.conversations.backend == "sql_default"
    assert stores.conversations.table_name == "openai_conversations"


def test_postgres_demo_distribution_config_loads():
    """Integration: Postgres demo should use Postgres backend for all stores."""
    with open("ogx/distributions/postgres-demo/config.yaml") as f:
        config_dict = yaml.safe_load(f)

    config = StackConfig(**config_dict)

    # Should have postgres backend
    assert config.storage is not None
    assert "kv_default" in config.storage.backends
    assert "sql_default" in config.storage.backends
    postgres_backend = config.storage.backends["sql_default"]
    assert isinstance(postgres_backend, PostgresSqlStoreConfig)
    assert postgres_backend.host == "${env.POSTGRES_HOST:=localhost}"

    kv_backend = config.storage.backends["kv_default"]
    assert isinstance(kv_backend, PostgresKVStoreConfig)

    stores = config.storage.stores
    # Stores target the Postgres backends explicitly
    assert stores.metadata is not None
    assert stores.metadata.backend == "kv_default"
    assert stores.inference is not None
    assert stores.inference.backend == "sql_default"
