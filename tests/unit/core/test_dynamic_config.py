# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""
Unit tests for run_config_from_dynamic_config_spec.

Categories:
  - Parsing: valid/invalid spec string formats (missing '=', semicolon separator)
  - Validation: unknown API name, API absent from registry, unknown provider
  - Provider resolution: exact match, inline:: prefix fallback, remote:: prefix fallback
  - Multi-provider: same API repeated accumulates providers instead of overwriting
  - Config shape: returned StackConfig has correct apis list, providers dict, and storage paths
"""

from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from ogx.core.datatypes import Api
from ogx.core.stack import run_config_from_dynamic_config_spec
from ogx_api import ProviderSpec


@pytest.fixture(autouse=True)
def _set_required_env_vars(monkeypatch: pytest.MonkeyPatch):
    """Provide defaults for env vars emitted by _SampleConfig.sample_run_config."""
    monkeypatch.setenv("MY_PROVIDER_API_KEY", "test-key")
    monkeypatch.delenv("MY_PROVIDER_URL", raising=False)
    monkeypatch.delenv("MY_OPTIONAL", raising=False)


class _SampleConfig(BaseModel):
    db_path: str = ""
    url: str = ""
    api_key: str = ""
    optional_field: str = ""

    @classmethod
    def sample_run_config(cls, __distro_dir__: str = "", **kwargs: Any) -> dict[str, Any]:
        return {
            "db_path": f"{__distro_dir__}/data.db",
            "url": "${env.MY_PROVIDER_URL:=http://localhost:8080}",
            "api_key": "${env.MY_PROVIDER_API_KEY}",
            "optional_field": "${env.MY_OPTIONAL:=}",
        }


def _make_registry(provider_type: str) -> dict[Api, dict[str, ProviderSpec]]:
    spec = ProviderSpec(
        api=Api.inference,
        provider_type=provider_type,
        config_class="tests.unit.core.test_dynamic_config._SampleConfig",
    )
    return {Api.inference: {provider_type: spec}}


class TestParsing:
    def test_missing_equals_raises(self):
        with pytest.raises(ValueError, match="Expected format"):
            run_config_from_dynamic_config_spec("inference", provider_registry={Api.inference: {}})

    def test_semicolon_separator_accepted(self, tmp_path: Path):
        registry = _make_registry("inline::test")
        config = run_config_from_dynamic_config_spec(
            "inference=inline::test;inference=inline::test",
            provider_registry=registry,
            distro_dir=tmp_path,
        )
        # both entries for the same API accumulate
        assert len(config.providers["inference"]) == 2

    def test_comma_separator_accepted(self, tmp_path: Path):
        registry = _make_registry("inline::test")
        config = run_config_from_dynamic_config_spec(
            "inference=inline::test,inference=inline::test",
            provider_registry=registry,
            distro_dir=tmp_path,
        )
        assert len(config.providers["inference"]) == 2


class TestValidation:
    def test_invalid_api_name_raises(self):
        with pytest.raises(ValueError, match="not a valid API"):
            run_config_from_dynamic_config_spec("notanapi=someprovider", provider_registry={})

    def test_api_absent_from_registry_raises(self):
        with pytest.raises(ValueError, match="Failed to find providers"):
            run_config_from_dynamic_config_spec("inference=fireworks", provider_registry={})

    def test_unknown_provider_raises(self):
        registry = {
            Api.inference: {
                "inline::other": ProviderSpec(
                    api=Api.inference,
                    provider_type="inline::other",
                    config_class="tests.unit.core.test_dynamic_config._SampleConfig",
                )
            }
        }
        with pytest.raises(ValueError, match="not found for API"):
            run_config_from_dynamic_config_spec("inference=fireworks", provider_registry=registry)


class TestProviderResolution:
    def test_exact_match(self, tmp_path: Path):
        registry = _make_registry("inline::fireworks")
        config = run_config_from_dynamic_config_spec(
            "inference=inline::fireworks",
            provider_registry=registry,
            distro_dir=tmp_path,
        )
        assert config.providers["inference"][0].provider_type == "inline::fireworks"

    def test_inline_prefix_fallback(self, tmp_path: Path):
        registry = _make_registry("inline::fireworks")
        config = run_config_from_dynamic_config_spec(
            "inference=fireworks",
            provider_registry=registry,
            distro_dir=tmp_path,
        )
        assert config.providers["inference"][0].provider_type == "inline::fireworks"

    def test_remote_prefix_fallback(self, tmp_path: Path):
        registry = _make_registry("remote::fireworks")
        config = run_config_from_dynamic_config_spec(
            "inference=fireworks",
            provider_registry=registry,
            distro_dir=tmp_path,
        )
        assert config.providers["inference"][0].provider_type == "remote::fireworks"


class TestMultiProvider:
    def test_same_api_twice_accumulates(self, tmp_path: Path):
        spec_a = ProviderSpec(
            api=Api.inference,
            provider_type="inline::a",
            config_class="tests.unit.core.test_dynamic_config._SampleConfig",
        )
        spec_b = ProviderSpec(
            api=Api.inference,
            provider_type="inline::b",
            config_class="tests.unit.core.test_dynamic_config._SampleConfig",
        )
        registry = {Api.inference: {"inline::a": spec_a, "inline::b": spec_b}}
        config = run_config_from_dynamic_config_spec(
            "inference=inline::a,inference=inline::b",
            provider_registry=registry,
            distro_dir=tmp_path,
        )
        provider_types = [p.provider_type for p in config.providers["inference"]]
        assert provider_types == ["inline::a", "inline::b"]


class TestEnvVarSubstitution:
    def test_default_value_used_when_env_not_set(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("MY_PROVIDER_URL", raising=False)
        monkeypatch.delenv("MY_OPTIONAL", raising=False)
        monkeypatch.setenv("MY_PROVIDER_API_KEY", "secret")
        config = run_config_from_dynamic_config_spec(
            "inference=inline::test",
            provider_registry=_make_registry("inline::test"),
            distro_dir=tmp_path,
        )
        provider_config = config.providers["inference"][0].config
        assert provider_config["url"] == "http://localhost:8080"
        assert provider_config["api_key"] == "secret"
        assert provider_config["optional_field"] is None

    def test_env_value_overrides_default(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("MY_PROVIDER_URL", "https://prod.example.com")
        monkeypatch.setenv("MY_PROVIDER_API_KEY", "prod-key")
        monkeypatch.delenv("MY_OPTIONAL", raising=False)
        config = run_config_from_dynamic_config_spec(
            "inference=inline::test",
            provider_registry=_make_registry("inline::test"),
            distro_dir=tmp_path,
        )
        provider_config = config.providers["inference"][0].config
        assert provider_config["url"] == "https://prod.example.com"
        assert provider_config["api_key"] == "prod-key"

    def test_missing_required_env_var_raises(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("MY_PROVIDER_API_KEY", raising=False)
        from ogx.core.stack import EnvVarError

        with pytest.raises(EnvVarError, match="MY_PROVIDER_API_KEY"):
            run_config_from_dynamic_config_spec(
                "inference=inline::test",
                provider_registry=_make_registry("inline::test"),
                distro_dir=tmp_path,
            )


class TestConfigShape:
    def test_apis_list_matches_providers(self, tmp_path: Path):
        registry = _make_registry("inline::test")
        config = run_config_from_dynamic_config_spec(
            "inference=inline::test",
            provider_registry=registry,
            distro_dir=tmp_path,
        )
        assert config.apis == ["inference"]
        assert "inference" in config.providers

    def test_storage_paths_use_distro_dir(self, tmp_path: Path):
        registry = _make_registry("inline::test")
        config = run_config_from_dynamic_config_spec(
            "inference=inline::test",
            provider_registry=registry,
            distro_dir=tmp_path,
        )
        backends = config.storage.backends
        # Paths are stored as env-var templates; distro_dir appears as the default value.
        assert f"SQLITE_STORE_DIR:={tmp_path}" in backends["kv_default"].db_path
        assert f"SQLITE_STORE_DIR:={tmp_path}" in backends["sql_default"].db_path

    def test_sqlite_store_dir_honoured(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("SQLITE_STORE_DIR", "/custom/sqlite/dir")
        registry = _make_registry("inline::test")
        config = run_config_from_dynamic_config_spec(
            "inference=inline::test",
            provider_registry=registry,
            distro_dir=tmp_path,
        )
        backends = config.storage.backends
        # The template is unresolved at this point — resolution happens when the config
        # is loaded at server-start time.  Verify the template is formed correctly and
        # that replace_env_vars resolves it to SQLITE_STORE_DIR.
        from ogx.core.stack import replace_env_vars

        resolved_kv = replace_env_vars(backends["kv_default"].db_path)
        resolved_sql = replace_env_vars(backends["sql_default"].db_path)
        assert resolved_kv == "/custom/sqlite/dir/kvstore.db"
        assert resolved_sql == "/custom/sqlite/dir/sql_store.db"

    def test_provider_id_set_to_spec_string(self, tmp_path: Path):
        registry = _make_registry("inline::test")
        config = run_config_from_dynamic_config_spec(
            "inference=inline::test",
            provider_registry=registry,
            distro_dir=tmp_path,
        )
        # provider_id strips the inline::/remote:: prefix so the routing table key is clean
        assert config.providers["inference"][0].provider_id == "test"
