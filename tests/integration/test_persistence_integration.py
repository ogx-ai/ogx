# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from pathlib import Path

import pytest
import yaml

import ogx
from ogx.core.datatypes import StackConfig
from ogx.core.stack import replace_env_vars
from ogx.core.storage.datatypes import (
    PostgresKVStoreConfig,
    PostgresSqlStoreConfig,
    SqliteKVStoreConfig,
    SqliteSqlStoreConfig,
)


def test_starter_distribution_config_loads_and_resolves():
    """Integration: Actual starter config should parse and have correct storage structure."""
    with open(Path(ogx.__file__).parent / "distributions" / "starter" / "config.yaml") as f:
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
    with open(Path(ogx.__file__).parent / "distributions" / "postgres-demo" / "config.yaml") as f:
        config_dict = yaml.safe_load(f)

    config = StackConfig(**config_dict)

    # Should have postgres backend
    assert config.storage is not None
    assert "kv_default" in config.storage.backends
    assert "sql_default" in config.storage.backends
    postgres_backend = config.storage.backends["sql_default"]
    assert isinstance(postgres_backend, PostgresSqlStoreConfig)
    assert postgres_backend.connection_string is None
    assert postgres_backend.host == "${env.POSTGRES_HOST:=localhost}"

    kv_backend = config.storage.backends["kv_default"]
    assert isinstance(kv_backend, PostgresKVStoreConfig)

    stores = config.storage.stores
    # Stores target the Postgres backends explicitly
    assert stores.metadata is not None
    assert stores.metadata.backend == "kv_default"
    assert stores.inference is not None
    assert stores.inference.backend == "sql_default"


def test_ci_postgres_example_keeps_component_fields() -> None:
    """CI's PostgreSQL example retains legacy fields to cover compatibility."""
    config_path = Path(ogx.__file__).parent / "distributions" / "ci-tests" / "run-with-postgres-store.yaml"
    with config_path.open() as f:
        config_dict = yaml.safe_load(f)

    kv_backend = config_dict["storage"]["backends"]["kv_default"]
    sql_backend = config_dict["storage"]["backends"]["sql_default"]
    assert kv_backend["type"] == "kv_postgres"
    assert sql_backend["type"] == "sql_postgres"
    for backend in (kv_backend, sql_backend):
        assert "connection_string" not in backend
        assert backend["host"] == "${env.POSTGRES_HOST:=localhost}"
        assert backend["user"] == "${env.POSTGRES_USER:=ogx}"


def test_starter_postgres_example_resolves_connection_string(monkeypatch: pytest.MonkeyPatch) -> None:
    config_path = Path(ogx.__file__).parent / "distributions" / "starter" / "run-with-postgres-uri-store.yaml"
    with config_path.open() as f:
        config_dict = yaml.safe_load(f)

    dsn = "postgresql://user:secret@db.example/app"
    monkeypatch.setenv("POSTGRES_CONNECTION_STRING", dsn)
    resolved_storage = replace_env_vars(config_dict["storage"])

    kv_backend = PostgresKVStoreConfig.model_validate(resolved_storage["backends"]["kv_default"])
    sql_backend = PostgresSqlStoreConfig.model_validate(resolved_storage["backends"]["sql_default"])
    assert kv_backend.connection_string is not None
    assert kv_backend.connection_string.get_secret_value() == dsn
    assert sql_backend.connection_string is not None
    assert sql_backend.connection_string.get_secret_value() == dsn


@pytest.mark.parametrize(
    "distribution, filename",
    [("starter", "run-with-postgres-store.yaml"), ("postgres-demo", "config.yaml")],
)
def test_postgres_examples_preserve_component_environment(
    monkeypatch: pytest.MonkeyPatch, distribution: str, filename: str
) -> None:
    config_path = Path(ogx.__file__).parent / "distributions" / distribution / filename
    with config_path.open() as f:
        config_dict = yaml.safe_load(f)

    monkeypatch.delenv("POSTGRES_CONNECTION_STRING", raising=False)
    monkeypatch.setenv("POSTGRES_HOST", "production-db.internal")
    monkeypatch.setenv("POSTGRES_USER", "production-user")
    monkeypatch.setenv("POSTGRES_PASSWORD", "production-secret")
    resolved_storage = replace_env_vars(config_dict["storage"])

    for backend_name, config_class in (
        ("kv_default", PostgresKVStoreConfig),
        ("sql_default", PostgresSqlStoreConfig),
    ):
        backend = config_class.model_validate(resolved_storage["backends"][backend_name])
        assert backend.connection_string is None
        assert backend.host == "production-db.internal"
        assert backend.user == "production-user"
        assert backend.password is not None
        assert backend.password.get_secret_value() == "production-secret"
