# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from types import SimpleNamespace
from typing import Any

from pydantic import BaseModel, Field

from ogx.core.datatypes import StackConfig
from ogx.core.resolver import instantiate_provider
from ogx.core.storage.datatypes import ServerStoresConfig, SqlStoreReference, StorageConfig
from ogx_api import Api, RemoteProviderSpec


class DummyRemoteConfig(BaseModel):
    metadata_store: SqlStoreReference | None = Field(default=None)


def _make_run_config(vector_store_table: str = "vector_store_metadata") -> StackConfig:
    return StackConfig(
        distro_name="test",
        providers={},
        storage=StorageConfig(
            stores=ServerStoresConfig(
                vector_stores=SqlStoreReference(
                    backend="sql_default",
                    table_name=vector_store_table,
                )
            )
        ),
    )


class TestResolverMetadataStoreInjection:
    async def test_injects_metadata_store_for_remote_provider_when_missing(self, monkeypatch):
        captured: dict[str, Any] = {}

        async def get_adapter_impl(config, deps, policy=None):
            captured["config"] = config
            return SimpleNamespace()

        monkeypatch.setattr("ogx.core.resolver.instantiate_class_type", lambda _: DummyRemoteConfig)
        monkeypatch.setattr(
            "ogx.core.resolver.importlib.import_module",
            lambda _: SimpleNamespace(get_adapter_impl=get_adapter_impl),
        )
        monkeypatch.setattr("ogx.core.resolver.check_protocol_compliance", lambda *_args, **_kwargs: None)

        provider = SimpleNamespace(
            provider_id="dummy-remote",
            provider_type="remote::dummy",
            config={},
            spec=RemoteProviderSpec(
                api=Api.vector_io,
                provider_type="remote::dummy",
                config_class="dummy.Config",
                module="dummy.remote.module",
                adapter_type="dummy-adapter",
            ),
        )

        await instantiate_provider(
            provider=provider,
            deps={},
            inner_impls={},
            dist_registry=SimpleNamespace(),
            run_config=_make_run_config(),
            policy=[],
        )

        config = captured["config"]
        assert config.metadata_store is not None
        assert config.metadata_store.backend == "sql_default"
        assert config.metadata_store.table_name == "vector_store_metadata"

    async def test_preserves_explicit_remote_metadata_store_config(self, monkeypatch):
        captured: dict[str, Any] = {}

        async def get_adapter_impl(config, deps, policy=None):
            captured["config"] = config
            return SimpleNamespace()

        monkeypatch.setattr("ogx.core.resolver.instantiate_class_type", lambda _: DummyRemoteConfig)
        monkeypatch.setattr(
            "ogx.core.resolver.importlib.import_module",
            lambda _: SimpleNamespace(get_adapter_impl=get_adapter_impl),
        )
        monkeypatch.setattr("ogx.core.resolver.check_protocol_compliance", lambda *_args, **_kwargs: None)

        provider = SimpleNamespace(
            provider_id="dummy-remote",
            provider_type="remote::dummy",
            config={
                "metadata_store": {
                    "backend": "sql_default",
                    "table_name": "custom_vector_store_table",
                }
            },
            spec=RemoteProviderSpec(
                api=Api.vector_io,
                provider_type="remote::dummy",
                config_class="dummy.Config",
                module="dummy.remote.module",
                adapter_type="dummy-adapter",
            ),
        )

        await instantiate_provider(
            provider=provider,
            deps={},
            inner_impls={},
            dist_registry=SimpleNamespace(),
            run_config=_make_run_config(vector_store_table="default_vector_store_table"),
            policy=[],
        )

        config = captured["config"]
        assert config.metadata_store is not None
        assert config.metadata_store.backend == "sql_default"
        assert config.metadata_store.table_name == "custom_vector_store_table"
