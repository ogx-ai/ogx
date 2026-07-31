# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Unit tests for storage backend/reference validation."""

import os

import pytest
from pydantic import ValidationError

from ogx.core.datatypes import (
    OGX_RUN_CONFIG_VERSION,
    StackConfig,
)
from ogx.core.storage.datatypes import (
    InferenceStoreReference,
    KVStoreReference,
    ServerStoresConfig,
    SqliteKVStoreConfig,
    SqliteSqlStoreConfig,
    SqlStoreReference,
    StorageConfig,
)


def _base_run_config(**overrides):
    metadata_reference = overrides.pop(
        "metadata_reference",
        KVStoreReference(backend="kv_default", namespace="registry"),
    )
    inference_reference = overrides.pop(
        "inference_reference",
        InferenceStoreReference(backend="sql_default", table_name="inference"),
    )
    conversations_reference = overrides.pop(
        "conversations_reference",
        SqlStoreReference(backend="sql_default", table_name="conversations"),
    )
    storage = overrides.pop(
        "storage",
        StorageConfig(
            backends={
                "kv_default": SqliteKVStoreConfig(db_path="/tmp/kv.db"),
                "sql_default": SqliteSqlStoreConfig(db_path="/tmp/sql.db"),
            },
            stores=ServerStoresConfig(
                metadata=metadata_reference,
                inference=inference_reference,
                conversations=conversations_reference,
            ),
        ),
    )
    return StackConfig(
        version=OGX_RUN_CONFIG_VERSION,
        distro_name="test-distro",
        apis=[],
        providers={},
        storage=storage,
        **overrides,
    )


def test_references_require_known_backend():
    with pytest.raises(ValidationError, match="unknown backend 'missing'"):
        _base_run_config(metadata_reference=KVStoreReference(backend="missing", namespace="registry"))


def test_references_must_match_backend_family():
    with pytest.raises(ValidationError, match="kv_.* is required"):
        _base_run_config(metadata_reference=KVStoreReference(backend="sql_default", namespace="registry"))

    with pytest.raises(ValidationError, match="sql_.* is required"):
        _base_run_config(
            inference_reference=InferenceStoreReference(backend="kv_default", table_name="inference"),
        )


def test_valid_configuration_passes_validation():
    config = _base_run_config()
    stores = config.storage.stores
    assert stores.metadata is not None and stores.metadata.backend == "kv_default"
    assert stores.inference is not None and stores.inference.backend == "sql_default"
    assert stores.conversations is not None and stores.conversations.backend == "sql_default"


@pytest.mark.parametrize("backend_key", ["kv_default", "sql_default"])
def test_default_backends_resolve_env_vars(backend_key, monkeypatch):
    """Default StorageConfig backends must contain real paths, not literal
    ``${env.SQLITE_STORE_DIR:=...}`` strings.  Pydantic defaults are set at
    object-construction time — before ``replace_env_vars()`` processes the
    YAML — so they must resolve environment variables eagerly.

    See https://github.com/ogx-ai/ogx/issues/4896
    """
    monkeypatch.delenv("SQLITE_STORE_DIR", raising=False)
    config = StorageConfig()
    db_path = config.backends[backend_key].db_path
    assert "${env." not in db_path, f"Unresolved env var syntax in default {backend_key}: {db_path}"


def test_default_backends_respect_sqlite_store_dir(monkeypatch):
    """When SQLITE_STORE_DIR is set, the default backends should use it."""
    monkeypatch.setenv("SQLITE_STORE_DIR", "/tmp/custom-store")
    config = StorageConfig()
    assert config.backends["kv_default"].db_path == "/tmp/custom-store/kvstore.db"
    assert config.backends["sql_default"].db_path == "/tmp/custom-store/sql_store.db"


def test_default_backends_fallback_to_distribs_base_dir(monkeypatch):
    """When SQLITE_STORE_DIR is unset, defaults should use DISTRIBS_BASE_DIR."""
    from ogx.core.utils.config_dirs import DISTRIBS_BASE_DIR

    monkeypatch.delenv("SQLITE_STORE_DIR", raising=False)
    config = StorageConfig()
    assert config.backends["kv_default"].db_path == os.path.join(str(DISTRIBS_BASE_DIR), "kvstore.db")
    assert config.backends["sql_default"].db_path == os.path.join(str(DISTRIBS_BASE_DIR), "sql_store.db")


def test_default_backends_expand_user_home(monkeypatch):
    """Tilde in SQLITE_STORE_DIR should be expanded to the user home directory."""
    monkeypatch.setenv("SQLITE_STORE_DIR", "~/custom-store")
    config = StorageConfig()
    for key in ("kv_default", "sql_default"):
        assert "~" not in config.backends[key].db_path
