# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from threading import Lock
from typing import Annotated, cast

from pydantic import Field

from ogx.core.storage.datatypes import (
    PostgresSqlStoreConfig,
    SqliteSqlStoreConfig,
    SqlStoreReference,
    StorageBackendConfig,
    StorageBackendType,
)
from ogx_api.internal.sqlstore import SqlStore

sql_store_pip_packages = ["sqlalchemy[asyncio]", "aiosqlite", "asyncpg"]

_SQLSTORE_BACKENDS: dict[str, StorageBackendConfig] = {}
_SQLSTORE_INSTANCES: dict[str, SqlStore] = {}
_SQLSTORE_LOCKS: dict[str, Lock] = {}


SqlStoreConfig = Annotated[
    SqliteSqlStoreConfig | PostgresSqlStoreConfig,
    Field(discriminator="type"),
]


def get_pip_packages(store_config: dict | SqlStoreConfig) -> list[str]:
    """Get pip packages for SQL store config, handling both dict and object cases."""
    if isinstance(store_config, dict):
        store_type = store_config.get("type")
        if store_type == StorageBackendType.SQL_SQLITE.value:
            return SqliteSqlStoreConfig.pip_packages()
        elif store_type == StorageBackendType.SQL_POSTGRES.value:
            return PostgresSqlStoreConfig.pip_packages()
        else:
            raise ValueError(f"Unknown SQL store type: {store_type}")
    else:
        return store_config.pip_packages()


def sqlstore_impl(reference: SqlStoreReference) -> SqlStore:
    """Get or create a SqlStore instance for the given store reference.

    Args:
        reference: A reference specifying the backend name and table name.

    Returns:
        A SqlStore instance for the referenced backend.

    Raises:
        ValueError: If the backend name is unknown or the backend type is unsupported.
    """
    backend_name = reference.backend

    backend_config = _SQLSTORE_BACKENDS.get(backend_name)
    if backend_config is None:
        raise ValueError(
            f"Unknown SQL store backend '{backend_name}'. Registered backends: {sorted(_SQLSTORE_BACKENDS)}"
        )

    existing = _SQLSTORE_INSTANCES.get(backend_name)
    if existing:
        return existing

    lock = _SQLSTORE_LOCKS.setdefault(backend_name, Lock())
    with lock:
        existing = _SQLSTORE_INSTANCES.get(backend_name)
        if existing:
            return existing

        if isinstance(backend_config, SqliteSqlStoreConfig | PostgresSqlStoreConfig):
            from .sqlalchemy_sqlstore import SqlAlchemySqlStoreImpl

            config = cast(SqliteSqlStoreConfig | PostgresSqlStoreConfig, backend_config).model_copy()
            instance = SqlAlchemySqlStoreImpl(config)
            _SQLSTORE_INSTANCES[backend_name] = instance
            return instance
        else:
            raise ValueError(f"Unknown sqlstore type {backend_config.type}")


def register_sqlstore_backends(backends: dict[str, StorageBackendConfig]) -> None:
    """Register the set of available SQL store backends for reference resolution."""
    global _SQLSTORE_BACKENDS
    global _SQLSTORE_INSTANCES

    _SQLSTORE_BACKENDS.clear()
    _SQLSTORE_INSTANCES.clear()
    _SQLSTORE_LOCKS.clear()
    for name, cfg in backends.items():
        _SQLSTORE_BACKENDS[name] = cfg


async def shutdown_sqlstore_backends() -> None:
    """Shutdown all cached SQL store instances."""
    global _SQLSTORE_INSTANCES
    for instance in _SQLSTORE_INSTANCES.values():
        await instance.shutdown()
    _SQLSTORE_INSTANCES.clear()
