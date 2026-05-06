# storage

Storage backends for OGX: key-value stores and SQL stores.

## Directory Structure

```text
storage/
  kvstore/             # Key-value store backends
    __init__.py
    config.py          # KVStore config classes
    kvstore.py         # KVStore factory and base implementation
    sqlite/            # SQLite KV backend (aiosqlite)
    redis/             # Redis KV backend
    postgres/          # PostgreSQL KV backend
    mongodb/           # MongoDB KV backend
  sqlstore/            # SQL store backends (SQLAlchemy-based)
    __init__.py
    sqlstore.py        # SqlStore factory and config
    sqlalchemy_sqlstore.py  # SQLAlchemy implementation
    authorized_sqlstore.py  # SqlStore with access control
  __init__.py
  datatypes.py         # Storage config types (StorageBackendType, KVStoreReference, etc.)
```

## KVStore

The `KVStore` interface provides simple key-value operations (`get`, `set`, `delete`, `keys`). Values are strings (typically JSON-serialized). Keys can be namespaced.

Backends: SQLite (default), Redis, PostgreSQL, MongoDB.

Used by: distribution registry, quota middleware, provider state persistence.

## SqlStore

The `SqlStore` interface provides typed table operations with column definitions, filtering, and pagination. Built on SQLAlchemy for portability.

Backends: SQLite (default), PostgreSQL.

Used by: inference store (chat completion logs), conversations, prompts.

## Configuration

Storage is configured in `StackConfig.storage` via `StorageConfig`. The `stores` field contains typed references (`KVStoreReference`, `SqlStoreReference`, `InferenceStoreReference`) that point to specific backend configurations.

See `datatypes.py` for all config types and `StorageBackendType` for the enum of supported backends.
