# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any
from unittest.mock import patch

import pytest
import yaml
from pydantic import BaseModel, Field, ValidationError

from ogx.core.datatypes import Api, Provider, StackConfig
from ogx.core.distribution import INTERNAL_APIS, get_provider_registry, providable_apis
from ogx.core.storage.datatypes import (
    InferenceStoreReference,
    KVStoreReference,
    ServerStoresConfig,
    SqliteKVStoreConfig,
    SqliteSqlStoreConfig,
    SqlStoreReference,
    StorageConfig,
)
from ogx_api import ProviderSpec


class SampleConfig(BaseModel):
    foo: str = Field(
        default="bar",
        description="foo",
    )

    @classmethod
    def sample_run_config(cls, **kwargs: Any) -> dict[str, Any]:
        return {
            "foo": "baz",
        }


def _default_storage() -> StorageConfig:
    return StorageConfig(
        backends={
            "kv_default": SqliteKVStoreConfig(db_path=":memory:"),
            "sql_default": SqliteSqlStoreConfig(db_path=":memory:"),
        },
        stores=ServerStoresConfig(
            metadata=KVStoreReference(backend="kv_default", namespace="registry"),
            inference=InferenceStoreReference(backend="sql_default", table_name="inference_store"),
            conversations=SqlStoreReference(backend="sql_default", table_name="conversations"),
            prompts=KVStoreReference(backend="kv_default", namespace="prompts"),
        ),
    )


def make_stack_config(**overrides) -> StackConfig:
    storage = overrides.pop("storage", _default_storage())
    defaults = dict(
        distro_name="test_image",
        apis=[],
        providers={},
        storage=storage,
    )
    defaults.update(overrides)
    return StackConfig(**defaults)


@pytest.fixture
def mock_providers():
    """Mock the available_providers function to return test providers."""
    with patch("ogx.providers.registry.inference.available_providers") as mock:
        mock.return_value = [
            ProviderSpec(
                provider_type="test_provider",
                api=Api.inference,
                adapter_type="test_adapter",
                config_class="test_provider.config.TestProviderConfig",
            )
        ]
        yield mock


@pytest.fixture
def base_config(tmp_path):
    """Create a base StackRunConfig with common settings."""
    return make_stack_config(
        apis=["inference"],
        providers={
            "inference": [
                Provider(
                    provider_id="sample_provider",
                    provider_type="sample",
                    config=SampleConfig.sample_run_config(),
                )
            ]
        },
        external_providers_dir=str(tmp_path),
    )


@pytest.fixture
def provider_spec_yaml():
    """Common provider spec YAML for testing."""
    return """
adapter_type: test_provider
config_class: test_provider.config.TestProviderConfig
module: test_provider
api_dependencies:
  - safety
"""


@pytest.fixture
def inline_provider_spec_yaml():
    """Common inline provider spec YAML for testing."""
    return """
module: test_provider
config_class: test_provider.config.TestProviderConfig
pip_packages:
  - test-package
api_dependencies:
  - safety
optional_api_dependencies:
  - vector_io
provider_data_validator: test_provider.validator.TestValidator
container_image: test-image:latest
"""


@pytest.fixture
def api_directories(tmp_path):
    """Create the API directory structure for testing."""
    # Create remote provider directory
    remote_inference_dir = tmp_path / "remote" / "inference"
    remote_inference_dir.mkdir(parents=True, exist_ok=True)

    # Create inline provider directory
    inline_inference_dir = tmp_path / "inline" / "inference"
    inline_inference_dir.mkdir(parents=True, exist_ok=True)

    return remote_inference_dir, inline_inference_dir


def make_import_module_side_effect(
    builtin_provider_spec=None,
    external_module=None,
    raise_for_external=False,
    missing_get_provider_spec=False,
):
    from types import SimpleNamespace

    def import_module_side_effect(name):
        if name == "ogx.providers.registry.inference":
            mock_builtin = SimpleNamespace(
                available_providers=lambda: [
                    builtin_provider_spec
                    or ProviderSpec(
                        api=Api.inference,
                        provider_type="test_provider",
                        config_class="test_provider.config.TestProviderConfig",
                        module="test_provider",
                    )
                ]
            )
            return mock_builtin
        elif name == "external_test.provider":
            if raise_for_external:
                raise ModuleNotFoundError(name)
            if missing_get_provider_spec:
                return SimpleNamespace()
            return external_module
        else:
            raise ModuleNotFoundError(name)

    return import_module_side_effect


class TestProviderRegistry:
    """Test suite for provider registry functionality."""

    def test_builtin_providers(self, mock_providers):
        """Test loading built-in providers."""
        registry = get_provider_registry(None)

        assert Api.inference in registry
        assert "test_provider" in registry[Api.inference]
        assert registry[Api.inference]["test_provider"].provider_type == "test_provider"
        assert registry[Api.inference]["test_provider"].api == Api.inference

    def test_internal_apis_excluded(self):
        """Test that internal APIs are excluded and APIs without provider registries are marked as internal."""
        import importlib

        apis = providable_apis()

        for internal_api in INTERNAL_APIS:
            assert internal_api not in apis, f"Internal API {internal_api} should not be in providable_apis"

        for api in apis:
            module_name = f"ogx.providers.registry.{api.name.lower()}"
            try:
                importlib.import_module(module_name)
            except ImportError as err:
                raise AssertionError(
                    f"API {api} is in providable_apis but has no provider registry module ({module_name})"
                ) from err

    def test_external_remote_providers(self, api_directories, mock_providers, base_config, provider_spec_yaml):
        """Test loading external remote providers from YAML files."""
        remote_dir, _ = api_directories
        with open(remote_dir / "test_provider.yaml", "w") as f:
            f.write(provider_spec_yaml)

        registry = get_provider_registry(base_config)
        assert len(registry[Api.inference]) == 2

        assert Api.inference in registry
        assert "remote::test_provider" in registry[Api.inference]
        provider = registry[Api.inference]["remote::test_provider"]
        assert provider.adapter_type == "test_provider"
        assert provider.module == "test_provider"
        assert provider.config_class == "test_provider.config.TestProviderConfig"
        assert Api.safety in provider.api_dependencies

    def test_external_inline_providers(self, api_directories, mock_providers, base_config, inline_provider_spec_yaml):
        """Test loading external inline providers from YAML files."""
        _, inline_dir = api_directories
        with open(inline_dir / "test_provider.yaml", "w") as f:
            f.write(inline_provider_spec_yaml)

        registry = get_provider_registry(base_config)
        assert len(registry[Api.inference]) == 2

        assert Api.inference in registry
        assert "inline::test_provider" in registry[Api.inference]
        provider = registry[Api.inference]["inline::test_provider"]
        assert provider.provider_type == "inline::test_provider"
        assert provider.module == "test_provider"
        assert provider.config_class == "test_provider.config.TestProviderConfig"
        assert provider.pip_packages == ["test-package"]
        assert Api.safety in provider.api_dependencies
        assert Api.vector_io in provider.optional_api_dependencies
        assert provider.provider_data_validator == "test_provider.validator.TestValidator"
        assert provider.container_image == "test-image:latest"

    def test_invalid_yaml(self, api_directories, mock_providers, base_config):
        """Test handling of invalid YAML files."""
        remote_dir, inline_dir = api_directories
        with open(remote_dir / "invalid.yaml", "w") as f:
            f.write("invalid: yaml: content: -")
        with open(inline_dir / "invalid.yaml", "w") as f:
            f.write("invalid: yaml: content: -")

        with pytest.raises(yaml.YAMLError):
            get_provider_registry(base_config)

    def test_missing_directory(self, mock_providers):
        """Test handling of missing external providers directory."""
        config = make_stack_config(
            apis=["inference"],
            providers={
                "inference": [
                    Provider(
                        provider_id="sample_provider",
                        provider_type="sample",
                        config=SampleConfig.sample_run_config(),
                    )
                ]
            },
            external_providers_dir="/nonexistent/dir",
        )
        with pytest.raises(FileNotFoundError):
            get_provider_registry(config=config)

    def test_empty_api_directory(self, api_directories, mock_providers, base_config):
        """Test handling of empty API directory."""
        registry = get_provider_registry(base_config)
        assert len(registry[Api.inference]) == 1  # Only built-in provider

    def test_malformed_remote_provider_spec(self, api_directories, mock_providers, base_config):
        """Test handling of malformed remote provider spec (missing required fields)."""
        remote_dir, _ = api_directories
        malformed_spec = """
adapter_type: test_provider
  # Missing required fields
api_dependencies:
  - safety
"""
        with open(remote_dir / "malformed.yaml", "w") as f:
            f.write(malformed_spec)

        with pytest.raises(ValidationError):
            get_provider_registry(base_config)

    def test_malformed_inline_provider_spec(self, api_directories, mock_providers, base_config):
        """Test handling of malformed inline provider spec (missing required fields)."""
        _, inline_dir = api_directories
        malformed_spec = """
module: test_provider
# Missing required config_class
pip_packages:
  - test-package
"""
        with open(inline_dir / "malformed.yaml", "w") as f:
            f.write(malformed_spec)

        with pytest.raises(ValidationError) as exc_info:
            get_provider_registry(base_config)
        assert "config_class" in str(exc_info.value)

    def test_external_provider_from_module_success(self, mock_providers):
        """Test loading an external provider from a module (success path)."""
        from types import SimpleNamespace

        from ogx_api import Api, ProviderSpec

        # Simulate a provider module with get_provider_spec
        fake_spec = ProviderSpec(
            api=Api.inference,
            provider_type="external_test",
            config_class="external_test.config.ExternalTestConfig",
            module="external_test",
        )
        fake_module = SimpleNamespace(get_provider_spec=lambda: fake_spec)

        import_module_side_effect = make_import_module_side_effect(external_module=fake_module)

        with patch("importlib.import_module", side_effect=import_module_side_effect) as mock_import:
            config = make_stack_config(
                distro_name="test_image",
                providers={
                    "inference": [
                        Provider(
                            provider_id="external_test",
                            provider_type="external_test",
                            config={},
                            module="external_test",
                        )
                    ]
                },
            )
            registry = get_provider_registry(config=config)
            assert Api.inference in registry
            assert "external_test" in registry[Api.inference]
            provider = registry[Api.inference]["external_test"]
            assert provider.module == "external_test"
            assert provider.config_class == "external_test.config.ExternalTestConfig"
            mock_import.assert_any_call("ogx.providers.registry.inference")
            mock_import.assert_any_call("external_test.provider")

    def test_external_provider_from_module_not_found(self, mock_providers):
        """Test handling ModuleNotFoundError for missing provider module."""

        import_module_side_effect = make_import_module_side_effect(raise_for_external=True)

        with patch("importlib.import_module", side_effect=import_module_side_effect):
            config = make_stack_config(
                distro_name="test_image",
                providers={
                    "inference": [
                        Provider(
                            provider_id="external_test",
                            provider_type="external_test",
                            config={},
                            module="external_test",
                        )
                    ]
                },
            )
            with pytest.raises(ValueError) as exc_info:
                get_provider_registry(config=config)
            assert "get_provider_spec not found" in str(exc_info.value)

    def test_external_provider_from_module_missing_get_provider_spec(self, mock_providers):
        """Test handling missing get_provider_spec in provider module (should raise ValueError)."""

        import_module_side_effect = make_import_module_side_effect(missing_get_provider_spec=True)

        with patch("importlib.import_module", side_effect=import_module_side_effect):
            config = make_stack_config(
                distro_name="test_image",
                providers={
                    "inference": [
                        Provider(
                            provider_id="external_test",
                            provider_type="external_test",
                            config={},
                            module="external_test",
                        )
                    ]
                },
            )
            with pytest.raises(AttributeError):
                get_provider_registry(config=config)

    def test_external_provider_from_module_listing(self, mock_providers):
        """Test loading an external provider from a module during list-deps (listing=True, partial spec)."""
        from ogx.core.datatypes import StackConfig
        from ogx_api import Api

        # No importlib patch needed, should not import module when listing
        config = StackConfig(
            distro_name="test_image",
            apis=[],
            providers={
                "inference": [
                    Provider(
                        provider_id="external_test",
                        provider_type="external_test",
                        config={},
                        module="external_test",
                    )
                ]
            },
        )
        registry = get_provider_registry(config=config, listing=True)
        assert Api.inference in registry
        assert "external_test" in registry[Api.inference]
        provider = registry[Api.inference]["external_test"]
        assert provider.module == "external_test"
        assert provider.is_external is True
        # config_class is empty string in partial spec
        assert provider.config_class == ""


class TestGetExternalProvidersFromModule:
    """Test suite for installing external providers from module."""

    def test_stackrunconfig_provider_without_module(self, mock_providers):
        """Test that providers without module attribute are skipped."""
        from ogx.core.distribution import get_external_providers_from_module

        import_module_side_effect = make_import_module_side_effect()

        with patch("importlib.import_module", side_effect=import_module_side_effect):
            config = make_stack_config(
                distro_name="test_image",
                providers={
                    "inference": [
                        Provider(
                            provider_id="no_module",
                            provider_type="no_module",
                            config={},
                        )
                    ]
                },
            )
            registry = {Api.inference: {}}
            result = get_external_providers_from_module(registry, config, listing=False)
            # Should not add anything to registry
            assert len(result[Api.inference]) == 0

    def test_stackrunconfig_with_version_spec(self, mock_providers):
        """Test provider with module containing version spec (e.g., package==1.0.0)."""
        from types import SimpleNamespace

        from ogx.core.distribution import get_external_providers_from_module
        from ogx_api import ProviderSpec

        fake_spec = ProviderSpec(
            api=Api.inference,
            provider_type="versioned_test",
            config_class="versioned_test.config.VersionedTestConfig",
            module="versioned_test==1.0.0",
        )
        fake_module = SimpleNamespace(get_provider_spec=lambda: fake_spec)

        def import_side_effect(name):
            if name == "versioned_test.provider":
                return fake_module
            raise ModuleNotFoundError(name)

        with patch("importlib.import_module", side_effect=import_side_effect):
            config = make_stack_config(
                distro_name="test_image",
                providers={
                    "inference": [
                        Provider(
                            provider_id="versioned",
                            provider_type="versioned_test",
                            config={},
                            module="versioned_test==1.0.0",
                        )
                    ]
                },
            )
            registry = {Api.inference: {}}
            result = get_external_providers_from_module(registry, config, listing=False)
            assert "versioned_test" in result[Api.inference]
            assert result[Api.inference]["versioned_test"].module == "versioned_test==1.0.0"

    def test_buildconfig_does_not_import_module(self, mock_providers):
        """Test that StackConfig does not import the module when listing (listing=True)."""
        from ogx.core.datatypes import StackConfig
        from ogx.core.distribution import get_external_providers_from_module

        config = StackConfig(
            distro_name="test_image",
            apis=[],
            providers={
                "inference": [
                    Provider(
                        provider_id="build_test",
                        provider_type="build_test",
                        config={},
                        module="build_test==1.0.0",
                    )
                ]
            },
        )

        # Should not call import_module at all when listing
        with patch("importlib.import_module") as mock_import:
            registry = {Api.inference: {}}
            result = get_external_providers_from_module(registry, config, listing=True)

            # Verify module was NOT imported
            mock_import.assert_not_called()

            # Verify partial spec was created
            assert "build_test" in result[Api.inference]
            provider = result[Api.inference]["build_test"]
            assert provider.module == "build_test==1.0.0"
            assert provider.is_external is True
            assert provider.config_class == ""
            assert provider.api == Api.inference

    def test_buildconfig_multiple_providers(self, mock_providers):
        """Test StackConfig with multiple providers for the same API."""
        from ogx.core.datatypes import StackConfig
        from ogx.core.distribution import get_external_providers_from_module

        config = StackConfig(
            distro_name="test_image",
            apis=[],
            providers={
                "inference": [
                    Provider(provider_id="provider1", provider_type="provider1", config={}, module="provider1"),
                    Provider(provider_id="provider2", provider_type="provider2", config={}, module="provider2"),
                ]
            },
        )

        with patch("importlib.import_module") as mock_import:
            registry = {Api.inference: {}}
            result = get_external_providers_from_module(registry, config, listing=True)

            mock_import.assert_not_called()
            assert "provider1" in result[Api.inference]
            assert "provider2" in result[Api.inference]

    def test_distributionspec_does_not_import_module(self, mock_providers):
        """Test that DistributionSpec does not import the module (listing=True)."""
        from ogx.core.datatypes import BuildProvider, DistributionSpec
        from ogx.core.distribution import get_external_providers_from_module

        dist_spec = DistributionSpec(
            description="test distribution",
            providers={
                "inference": [
                    BuildProvider(
                        provider_type="dist_test",
                        module="dist_test==2.0.0",
                    )
                ]
            },
        )

        # Should not call import_module at all when listing
        with patch("importlib.import_module") as mock_import:
            registry = {Api.inference: {}}
            result = get_external_providers_from_module(registry, dist_spec, listing=True)

            # Verify module was NOT imported
            mock_import.assert_not_called()

            # Verify partial spec was created
            assert "dist_test" in result[Api.inference]
            provider = result[Api.inference]["dist_test"]
            assert provider.module == "dist_test==2.0.0"
            assert provider.is_external is True
            assert provider.config_class == ""

    def test_list_return_from_get_provider_spec(self, mock_providers):
        """Test when get_provider_spec returns a list of specs."""
        from types import SimpleNamespace

        from ogx.core.distribution import get_external_providers_from_module
        from ogx_api import ProviderSpec

        spec1 = ProviderSpec(
            api=Api.inference,
            provider_type="list_test",
            config_class="list_test.config.Config1",
            module="list_test",
        )
        spec2 = ProviderSpec(
            api=Api.inference,
            provider_type="list_test_remote",
            config_class="list_test.config.Config2",
            module="list_test",
        )

        fake_module = SimpleNamespace(get_provider_spec=lambda: [spec1, spec2])

        def import_side_effect(name):
            if name == "list_test.provider":
                return fake_module
            raise ModuleNotFoundError(name)

        with patch("importlib.import_module", side_effect=import_side_effect):
            config = make_stack_config(
                distro_name="test_image",
                providers={
                    "inference": [
                        Provider(
                            provider_id="list_test",
                            provider_type="list_test",
                            config={},
                            module="list_test",
                        )
                    ]
                },
            )
            registry = {Api.inference: {}}
            result = get_external_providers_from_module(registry, config, listing=False)

            # Only the matching provider_type should be added
            assert "list_test" in result[Api.inference]
            assert result[Api.inference]["list_test"].config_class == "list_test.config.Config1"

    def test_list_return_filters_by_provider_type(self, mock_providers):
        """Test that list return filters specs by provider_type."""
        from types import SimpleNamespace

        from ogx.core.distribution import get_external_providers_from_module
        from ogx_api import ProviderSpec

        spec1 = ProviderSpec(
            api=Api.inference,
            provider_type="wanted",
            config_class="test.Config1",
            module="test",
        )
        spec2 = ProviderSpec(
            api=Api.inference,
            provider_type="unwanted",
            config_class="test.Config2",
            module="test",
        )

        fake_module = SimpleNamespace(get_provider_spec=lambda: [spec1, spec2])

        def import_side_effect(name):
            if name == "test.provider":
                return fake_module
            raise ModuleNotFoundError(name)

        with patch("importlib.import_module", side_effect=import_side_effect):
            config = make_stack_config(
                distro_name="test_image",
                providers={
                    "inference": [
                        Provider(
                            provider_id="wanted",
                            provider_type="wanted",
                            config={},
                            module="test",
                        )
                    ]
                },
            )
            registry = {Api.inference: {}}
            result = get_external_providers_from_module(registry, config, listing=False)

            # Only the matching provider_type should be added
            assert "wanted" in result[Api.inference]
            assert "unwanted" not in result[Api.inference]

    def test_list_return_adds_multiple_provider_types(self, mock_providers):
        """Test that list return adds multiple different provider_types when config requests them."""
        from types import SimpleNamespace

        from ogx.core.distribution import get_external_providers_from_module
        from ogx_api import ProviderSpec

        # Module returns both inline and remote variants
        spec1 = ProviderSpec(
            api=Api.inference,
            provider_type="remote::ollama",
            config_class="test.RemoteConfig",
            module="test",
        )
        spec2 = ProviderSpec(
            api=Api.inference,
            provider_type="inline::ollama",
            config_class="test.InlineConfig",
            module="test",
        )

        fake_module = SimpleNamespace(get_provider_spec=lambda: [spec1, spec2])

        def import_side_effect(name):
            if name == "test.provider":
                return fake_module
            raise ModuleNotFoundError(name)

        with patch("importlib.import_module", side_effect=import_side_effect):
            config = make_stack_config(
                distro_name="test_image",
                providers={
                    "inference": [
                        Provider(
                            provider_id="remote_ollama",
                            provider_type="remote::ollama",
                            config={},
                            module="test",
                        ),
                        Provider(
                            provider_id="inline_ollama",
                            provider_type="inline::ollama",
                            config={},
                            module="test",
                        ),
                    ]
                },
            )
            registry = {Api.inference: {}}
            result = get_external_providers_from_module(registry, config, listing=False)

            # Both provider types should be added to registry
            assert "remote::ollama" in result[Api.inference]
            assert "inline::ollama" in result[Api.inference]
            assert result[Api.inference]["remote::ollama"].config_class == "test.RemoteConfig"
            assert result[Api.inference]["inline::ollama"].config_class == "test.InlineConfig"

    def test_module_not_found_raises_value_error(self, mock_providers):
        """Test that ModuleNotFoundError raises ValueError with helpful message."""
        from ogx.core.distribution import get_external_providers_from_module

        def import_side_effect(name):
            if name == "missing_module.provider":
                raise ModuleNotFoundError(name)
            raise ModuleNotFoundError(name)

        with patch("importlib.import_module", side_effect=import_side_effect):
            config = make_stack_config(
                distro_name="test_image",
                providers={
                    "inference": [
                        Provider(
                            provider_id="missing",
                            provider_type="missing",
                            config={},
                            module="missing_module",
                        )
                    ]
                },
            )
            registry = {Api.inference: {}}

            with pytest.raises(ValueError) as exc_info:
                get_external_providers_from_module(registry, config, listing=False)

            assert "get_provider_spec not found" in str(exc_info.value)

    def test_generic_exception_is_raised(self, mock_providers):
        """Test that generic exceptions are properly raised."""
        from types import SimpleNamespace

        from ogx.core.distribution import get_external_providers_from_module

        def bad_spec():
            raise RuntimeError("Something went wrong")

        fake_module = SimpleNamespace(get_provider_spec=bad_spec)

        def import_side_effect(name):
            if name == "error_module.provider":
                return fake_module
            raise ModuleNotFoundError(name)

        with patch("importlib.import_module", side_effect=import_side_effect):
            config = make_stack_config(
                distro_name="test_image",
                providers={
                    "inference": [
                        Provider(
                            provider_id="error",
                            provider_type="error",
                            config={},
                            module="error_module",
                        )
                    ]
                },
            )
            registry = {Api.inference: {}}

            with pytest.raises(RuntimeError) as exc_info:
                get_external_providers_from_module(registry, config, listing=False)

            assert "Something went wrong" in str(exc_info.value)

    def test_empty_provider_list(self, mock_providers):
        """Test with empty provider list."""
        from ogx.core.distribution import get_external_providers_from_module

        config = make_stack_config(
            distro_name="test_image",
            providers={},
        )
        registry = {Api.inference: {}}
        result = get_external_providers_from_module(registry, config, listing=False)

        # Should return registry unchanged
        assert result == registry
        assert len(result[Api.inference]) == 0

    def test_multiple_apis_with_providers(self, mock_providers):
        """Test multiple APIs with providers."""
        from types import SimpleNamespace

        from ogx.core.distribution import get_external_providers_from_module
        from ogx_api import ProviderSpec

        inference_spec = ProviderSpec(
            api=Api.inference,
            provider_type="inf_test",
            config_class="inf.Config",
            module="inf_test",
        )
        safety_spec = ProviderSpec(
            api=Api.safety,
            provider_type="safe_test",
            config_class="safe.Config",
            module="safe_test",
        )

        def import_side_effect(name):
            if name == "inf_test.provider":
                return SimpleNamespace(get_provider_spec=lambda: inference_spec)
            elif name == "safe_test.provider":
                return SimpleNamespace(get_provider_spec=lambda: safety_spec)
            raise ModuleNotFoundError(name)

        with patch("importlib.import_module", side_effect=import_side_effect):
            config = make_stack_config(
                distro_name="test_image",
                providers={
                    "inference": [
                        Provider(
                            provider_id="inf",
                            provider_type="inf_test",
                            config={},
                            module="inf_test",
                        )
                    ],
                    "safety": [
                        Provider(
                            provider_id="safe",
                            provider_type="safe_test",
                            config={},
                            module="safe_test",
                        )
                    ],
                },
            )
            registry = {Api.inference: {}, Api.safety: {}}
            result = get_external_providers_from_module(registry, config, listing=False)

            assert "inf_test" in result[Api.inference]
            assert "safe_test" in result[Api.safety]
