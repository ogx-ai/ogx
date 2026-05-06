# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import os
import re
from abc import abstractmethod
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, Field, field_validator

from ogx.core.utils.config_dirs import DISTRIBS_BASE_DIR


class StorageBackendType(StrEnum):
    """Supported storage backend types for key-value and SQL stores."""

    KV_REDIS = "kv_redis"
    KV_SQLITE = "kv_sqlite"
    KV_POSTGRES = "kv_postgres"
    KV_MONGODB = "kv_mongodb"
    SQL_SQLITE = "sql_sqlite"
    SQL_POSTGRES = "sql_postgres"


class CommonConfig(BaseModel):
    """Base configuration shared by all key-value store backends."""

    namespace: str | None = Field(
        default=None,
        description="All keys will be prefixed with this namespace",
    )


class RedisKVStoreConfig(CommonConfig):
    """Configuration for the Redis key-value store backend."""

    type: Literal[StorageBackendType.KV_REDIS] = StorageBackendType.KV_REDIS
    host: str = "localhost"
    port: int = 6379

    @property
    def url(self) -> str:
        return f"redis://{self.host}:{self.port}"

    @classmethod
    def pip_packages(cls) -> list[str]:
        return ["redis"]

    @classmethod
    def sample_run_config(cls) -> dict[str, str]:
        return {
            "type": StorageBackendType.KV_REDIS.value,
            "host": "${env.REDIS_HOST:=localhost}",
            "port": "${env.REDIS_PORT:=6379}",
        }


class SqliteKVStoreConfig(CommonConfig):
    """Configuration for the SQLite key-value store backend."""

    type: Literal[StorageBackendType.KV_SQLITE] = StorageBackendType.KV_SQLITE
    db_path: str = Field(
        description="File path for the sqlite database",
    )

    @classmethod
    def pip_packages(cls) -> list[str]:
        return ["aiosqlite"]

    @classmethod
    def sample_run_config(cls, __distro_dir__: str, db_name: str = "kvstore.db") -> dict[str, str]:
        return {
            "type": StorageBackendType.KV_SQLITE.value,
            "db_path": "${env.SQLITE_STORE_DIR:=" + __distro_dir__ + "}/" + db_name,
        }


class PostgresKVStoreConfig(CommonConfig):
    """Configuration for the PostgreSQL key-value store backend."""

    type: Literal[StorageBackendType.KV_POSTGRES] = StorageBackendType.KV_POSTGRES
    host: str = "localhost"
    port: int | str = 5432
    db: str = "ogx"
    user: str
    password: str | None = None
    ssl_mode: str | None = None
    ca_cert_path: str | None = None
    table_name: str = "ogx_kvstore"

    @classmethod
    def sample_run_config(cls, table_name: str = "ogx_kvstore", **kwargs: object) -> dict[str, str]:
        return {
            "type": StorageBackendType.KV_POSTGRES.value,
            "host": "${env.POSTGRES_HOST:=localhost}",
            "port": "${env.POSTGRES_PORT:=5432}",
            "db": "${env.POSTGRES_DB:=ogx}",
            "user": "${env.POSTGRES_USER:=ogx}",
            "password": "${env.POSTGRES_PASSWORD:=ogx}",
            "table_name": "${env.POSTGRES_TABLE_NAME:=" + table_name + "}",
        }

    @classmethod
    @field_validator("table_name")
    def validate_table_name(cls, v: str) -> str:
        # PostgreSQL identifiers rules:
        # - Must start with a letter or underscore
        # - Can contain letters, numbers, and underscores
        # - Maximum length is 63 bytes
        pattern = r"^[a-zA-Z_][a-zA-Z0-9_]*$"
        if not re.match(pattern, v):
            raise ValueError(
                "Invalid table name. Must start with letter or underscore and contain only letters, numbers, and underscores"
            )
        if len(v) > 63:
            raise ValueError("Table name must be less than 63 characters")
        return v

    @classmethod
    def pip_packages(cls) -> list[str]:
        return ["psycopg2-binary"]


class MongoDBKVStoreConfig(CommonConfig):
    """Configuration for the MongoDB key-value store backend."""

    type: Literal[StorageBackendType.KV_MONGODB] = StorageBackendType.KV_MONGODB
    host: str = "localhost"
    port: int = 27017
    db: str = "ogx"
    user: str | None = None
    password: str | None = None
    collection_name: str = "ogx_kvstore"

    @classmethod
    def pip_packages(cls) -> list[str]:
        return ["pymongo"]

    @classmethod
    def sample_run_config(cls, collection_name: str = "ogx_kvstore") -> dict[str, str]:
        return {
            "type": StorageBackendType.KV_MONGODB.value,
            "host": "${env.MONGODB_HOST:=localhost}",
            "port": "${env.MONGODB_PORT:=5432}",
            "db": "${env.MONGODB_DB}",
            "user": "${env.MONGODB_USER}",
            "password": "${env.MONGODB_PASSWORD}",
            "collection_name": "${env.MONGODB_COLLECTION_NAME:=" + collection_name + "}",
        }


class SqlAlchemySqlStoreConfig(BaseModel):
    """Base configuration for SQLAlchemy-based SQL store backends."""

    pool_pre_ping: bool = True

    @property
    @abstractmethod
    def engine_str(self) -> str: ...

    # TODO: move this when we have a better way to specify dependencies with internal APIs
    @classmethod
    def pip_packages(cls) -> list[str]:
        return ["sqlalchemy[asyncio]"]


class SqliteSqlStoreConfig(SqlAlchemySqlStoreConfig):
    """Configuration for the SQLite SQL store backend."""

    type: Literal[StorageBackendType.SQL_SQLITE] = StorageBackendType.SQL_SQLITE
    db_path: str = Field(
        description="Database path, e.g. ~/.ogx/distributions/ollama/sqlstore.db",
    )

    @property
    def engine_str(self) -> str:
        return "sqlite+aiosqlite:///" + Path(self.db_path).expanduser().as_posix()

    @classmethod
    def sample_run_config(cls, __distro_dir__: str, db_name: str = "sqlstore.db") -> dict[str, str]:
        return {
            "type": StorageBackendType.SQL_SQLITE.value,
            "db_path": "${env.SQLITE_STORE_DIR:=" + __distro_dir__ + "}/" + db_name,
        }

    @classmethod
    def pip_packages(cls) -> list[str]:
        return super().pip_packages() + ["aiosqlite"]


class PostgresSqlStoreConfig(SqlAlchemySqlStoreConfig):
    """Configuration for the PostgreSQL SQL store backend."""

    type: Literal[StorageBackendType.SQL_POSTGRES] = StorageBackendType.SQL_POSTGRES
    host: str = "localhost"
    port: int | str = 5432
    db: str = "ogx"
    user: str
    password: str | None = None
    pool_size: int = Field(default=10, ge=1, description="Number of persistent connections in the pool")
    max_overflow: int = Field(default=20, ge=0, description="Max additional connections beyond pool_size")
    pool_recycle: int = Field(default=-1, ge=-1, description="Connection recycle interval in seconds, -1 to disable")

    @property
    def engine_str(self) -> str:
        return f"postgresql+asyncpg://{self.user}:{self.password}@{self.host}:{self.port}/{self.db}"

    @classmethod
    def pip_packages(cls) -> list[str]:
        return super().pip_packages() + ["asyncpg"]

    @classmethod
    def sample_run_config(cls, **kwargs: object) -> dict[str, str]:
        return {
            "type": StorageBackendType.SQL_POSTGRES.value,
            "host": "${env.POSTGRES_HOST:=localhost}",
            "port": "${env.POSTGRES_PORT:=5432}",
            "db": "${env.POSTGRES_DB:=ogx}",
            "user": "${env.POSTGRES_USER:=ogx}",
            "password": "${env.POSTGRES_PASSWORD:=ogx}",
            "pool_size": "${env.POSTGRES_POOL_SIZE:=10}",
            "max_overflow": "${env.POSTGRES_MAX_OVERFLOW:=20}",
            "pool_recycle": "${env.POSTGRES_POOL_RECYCLE:=-1}",
            "pool_pre_ping": "${env.POSTGRES_POOL_PRE_PING:=true}",
        }


# reference = (backend_name, table_name)
class SqlStoreReference(BaseModel):
    """A reference to a 'SQL-like' persistent store. A table name must be provided."""

    table_name: str = Field(
        description="Name of the table to use for the SqlStore",
    )

    backend: str = Field(
        description="Name of backend from storage.backends",
    )


# reference = (backend_name, namespace)
class KVStoreReference(BaseModel):
    """A reference to a 'key-value' persistent store. A namespace must be provided."""

    namespace: str = Field(
        description="Key prefix for KVStore backends",
    )

    backend: str = Field(
        description="Name of backend from storage.backends",
    )


StorageBackendConfig = Annotated[
    RedisKVStoreConfig
    | SqliteKVStoreConfig
    | PostgresKVStoreConfig
    | MongoDBKVStoreConfig
    | SqliteSqlStoreConfig
    | PostgresSqlStoreConfig,
    Field(discriminator="type"),
]


class InferenceStoreReference(SqlStoreReference):
    """Inference store configuration with queue tuning."""

    max_write_queue_size: int = Field(
        default=10000,
        description="Max queued writes for inference store",
    )
    num_writers: int = Field(
        default=4,
        description="Number of concurrent background writers",
    )


class ResponsesStoreReference(InferenceStoreReference):
    """Responses store configuration with queue tuning."""

    table_name: str = Field(
        default="openai_responses",
        description="Name of the table to use for storing OpenAI responses",
    )


class ServerStoresConfig(BaseModel):
    """Configuration mapping logical store names to their backend references."""

    metadata: KVStoreReference | None = Field(
        default=KVStoreReference(
            backend="kv_default",
            namespace="registry",
        ),
        description="Metadata store configuration (uses KV backend)",
    )
    inference: InferenceStoreReference | None = Field(
        default=InferenceStoreReference(
            backend="sql_default",
            table_name="inference_store",
        ),
        description="Inference store configuration (uses SQL backend)",
    )
    conversations: SqlStoreReference | None = Field(
        default=SqlStoreReference(
            backend="sql_default",
            table_name="openai_conversations",
        ),
        description="Conversations store configuration (uses SQL backend)",
    )
    responses: ResponsesStoreReference | None = Field(
        default=None,
        description="Responses store configuration (uses SQL backend)",
    )
    prompts: KVStoreReference | None = Field(
        default=KVStoreReference(backend="kv_default", namespace="prompts"),
        description="Prompts store configuration (uses KV backend)",
    )
    connectors: KVStoreReference | None = Field(
        default=KVStoreReference(backend="kv_default", namespace="connectors"),
        description="Connectors store configuration (uses KV backend)",
    )


def _default_backends() -> dict[str, StorageBackendConfig]:
    base_dir = os.path.expanduser(os.environ.get("SQLITE_STORE_DIR") or str(DISTRIBS_BASE_DIR))
    return {
        "kv_default": SqliteKVStoreConfig(
            db_path=os.path.join(base_dir, "kvstore.db"),
        ),
        "sql_default": SqliteSqlStoreConfig(
            db_path=os.path.join(base_dir, "sql_store.db"),
        ),
    }


class StorageConfig(BaseModel):
    """Top-level storage configuration defining backends and store references."""

    # default_factory resolves SQLITE_STORE_DIR at construction time via
    # os.environ.get() instead of embedding literal ${env.SQLITE_STORE_DIR:=...}
    # strings that would bypass replace_env_vars() and crash on read-only
    # container filesystems.  See https://github.com/ogx-ai/ogx/issues/4896
    backends: dict[str, StorageBackendConfig] = Field(
        default_factory=_default_backends,
        description="Named backend configurations (e.g., 'default', 'cache')",
    )
    stores: ServerStoresConfig = Field(
        default_factory=lambda: ServerStoresConfig(),
        description="Named references to storage backends used by the stack core",
    )
