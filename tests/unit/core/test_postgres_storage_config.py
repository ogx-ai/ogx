# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Validation and display behavior for PostgreSQL storage connection URIs."""

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from pydantic import SecretStr

from ogx.core.admin import AdminImpl, AdminImplConfig
from ogx.core.datatypes import Provider, StackConfig
from ogx.core.exceptions.translation import translate_exception
from ogx.core.providers import ProviderImpl, ProviderImplConfig
from ogx.core.storage.datatypes import PostgresKVStoreConfig, PostgresSqlStoreConfig
from ogx.core.utils.config import redact_sensitive_fields

POSTGRES_CONFIGS = [PostgresKVStoreConfig, PostgresSqlStoreConfig]
_COMPONENT_FIELDS = {"host", "port", "db", "user", "password"}


@pytest.mark.parametrize("config_class", POSTGRES_CONFIGS)
@pytest.mark.parametrize(
    "connection_string",
    ["postgresql+asyncpg://user:secret@db/app", "not-a-uri"],
)
def test_connection_string_rejects_unsupported_scheme(config_class: Any, connection_string: str) -> None:
    with pytest.raises(ValueError, match="postgres://.*postgresql://"):
        config_class(connection_string=connection_string)


@pytest.mark.parametrize("config_class", POSTGRES_CONFIGS)
@pytest.mark.parametrize("scheme", ["postgres://", "postgresql://"])
def test_connection_string_accepts_postgres_schemes(config_class: Any, scheme: str) -> None:
    config = config_class(connection_string=f"{scheme}user:secret@db/app")

    assert config.connection_string is not None
    assert config.connection_string.get_secret_value() == f"{scheme}user:secret@db/app"


@pytest.mark.parametrize("config_class", POSTGRES_CONFIGS)
@pytest.mark.parametrize(
    "connection_string",
    [
        "postgresql://user:secret@",
        "postgresql://user:secret@/app",
        "postgresql://user:secret@db/",
    ],
)
def test_connection_string_requires_host_and_database_path(config_class: Any, connection_string: str) -> None:
    with pytest.raises(ValueError, match="host and database path"):
        config_class(connection_string=connection_string)


@pytest.mark.parametrize("config_class", POSTGRES_CONFIGS)
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("host", "localhost"),
        ("port", 5432),
        ("db", "ogx"),
        ("user", "other-user"),
        ("password", SecretStr("other-password")),
    ],
)
def test_connection_string_rejects_explicit_component_fields(
    config_class: Any, field: str, value: str | int | SecretStr
) -> None:
    with pytest.raises(ValueError, match="component fields"):
        config_class(connection_string="postgresql://user:secret@db/app", **{field: value})


@pytest.mark.parametrize("config_class", POSTGRES_CONFIGS)
@pytest.mark.parametrize(
    ("uri_option", "config_field", "config_value"),
    [
        ("sslmode=require", "ssl_mode", "require"),
        ("sslrootcert=%2Fetc%2Fpostgres%2Fca.pem", "ca_cert_path", "/etc/postgres/ca.pem"),
    ],
)
def test_connection_string_rejects_duplicate_ssl_settings(
    config_class: Any, uri_option: str, config_field: str, config_value: str
) -> None:
    connection_string = f"postgresql://user:secret@db/app?{uri_option}"

    with pytest.raises(ValueError, match="PostgreSQL SSL"):
        config_class(connection_string=connection_string, **{config_field: config_value})


@pytest.mark.parametrize("config_class", POSTGRES_CONFIGS)
def test_postgres_config_requires_connection_string_or_user(config_class: Any) -> None:
    with pytest.raises(ValueError, match="connection_string or user"):
        config_class()


@pytest.mark.parametrize("config_class", POSTGRES_CONFIGS)
def test_connection_string_round_trip_omits_default_component_fields(config_class: Any) -> None:
    config = config_class(connection_string="postgresql://user:secret@db/app")

    serialized_config = config.model_dump()
    assert not _COMPONENT_FIELDS.intersection(serialized_config)
    restored_config = config_class.model_validate(serialized_config)
    assert restored_config.connection_string is not None
    assert restored_config.connection_string.get_secret_value() == "postgresql://user:secret@db/app"


def test_connection_string_is_redacted_from_displayed_config() -> None:
    config = PostgresSqlStoreConfig(connection_string="postgresql://user:top-secret@db/app")
    displayed_config = redact_sensitive_fields(config.model_dump(mode="json"))

    assert displayed_config["connection_string"] == "********"
    assert "top-secret" not in str(displayed_config)


@pytest.mark.parametrize("config_class", POSTGRES_CONFIGS)
def test_connection_string_validation_error_hides_credentials(config_class: Any) -> None:
    password = "top-secret"
    with pytest.raises(ValueError) as exc_info:
        config_class(connection_string=f"postgresql://user:{password}@[invalid/db")

    assert password not in str(exc_info.value)
    assert password not in str(translate_exception(exc_info.value).detail)


@pytest.mark.parametrize("config_class", POSTGRES_CONFIGS)
def test_connection_string_accepts_multiple_hosts(config_class: Any) -> None:
    dsn = "postgresql://user:secret@db1.example:5432,db2.example:5433/app"

    config = config_class(connection_string=dsn)

    assert config.connection_string is not None
    assert config.connection_string.get_secret_value() == dsn


@pytest.mark.parametrize("impl_class, config_class", [(AdminImpl, AdminImplConfig), (ProviderImpl, ProviderImplConfig)])
async def test_provider_listing_redacts_without_revalidating_storage(impl_class: Any, config_class: Any) -> None:
    stack_config = StackConfig(
        distro_name="test",
        providers={
            "inference": [
                Provider(provider_id="test", provider_type="remote::test", config={"api_key": "provider-secret"})
            ]
        },
        storage={
            "backends": {
                "kv_default": {"type": "kv_postgres", "connection_string": "postgresql://user:secret@db/app"},
                "sql_default": {"type": "sql_postgres", "connection_string": "postgresql://user:secret@db/app"},
            }
        },
    )
    impl = impl_class(config_class(config=stack_config), {})

    with patch.object(impl, "get_providers_health", new_callable=AsyncMock, return_value={}):
        response = await impl.list_providers()

    assert response.data[0].config["api_key"] == "********"


def test_stack_config_validation_error_hides_connection_string_credentials() -> None:
    with pytest.raises(ValueError) as exc_info:
        StackConfig(
            distro_name="test",
            providers={},
            storage={
                "backends": {
                    "kv_default": {"type": "kv_postgres", "connection_string": "postgresql://user:top-secret@[bad/db"},
                    "sql_default": {"type": "sql_postgres", "connection_string": "postgresql://user:top-secret@db/app"},
                }
            },
        )

    assert "top-secret" not in str(exc_info.value)
