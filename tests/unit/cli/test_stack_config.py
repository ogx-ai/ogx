# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from datetime import datetime

import pytest
import yaml

from ogx.core.configure import (
    OGX_RUN_CONFIG_VERSION,
    parse_and_maybe_upgrade_config,
)


@pytest.fixture
def config_with_distro_name_int():
    return yaml.safe_load(
        f"""
        version: {OGX_RUN_CONFIG_VERSION}
        distro_name: 1234
        apis_to_serve: []
        built_at: {datetime.now().isoformat()}
        storage:
          backends:
            kv_default:
              type: kv_sqlite
              db_path: /tmp/test_kv.db
            sql_default:
              type: sql_sqlite
              db_path: /tmp/test_sql.db
          stores:
            metadata:
              backend: kv_default
              namespace: metadata
            inference:
              backend: sql_default
              table_name: inference
            conversations:
              backend: sql_default
              table_name: conversations
            responses:
              backend: sql_default
              table_name: responses
            prompts:
              backend: kv_default
              namespace: prompts
        providers:
          inference:
            - provider_id: provider1
              provider_type: remote::ollama
              config: {{}}
          safety:
            - provider_id: provider1
              provider_type: inline::llama-guard
              config:
                llama_guard_shield:
                  model: Llama-Guard-3-1B
                  excluded_categories: []
                  disable_input_check: false
                  disable_output_check: false
                enable_prompt_guard: false
          memory:
            - provider_id: provider1
              provider_type: inline::builtin
              config: {{}}
    """
    )


@pytest.fixture
def up_to_date_config():
    return yaml.safe_load(
        f"""
        version: {OGX_RUN_CONFIG_VERSION}
        distro_name: foo
        apis_to_serve: []
        built_at: {datetime.now().isoformat()}
        storage:
          backends:
            kv_default:
              type: kv_sqlite
              db_path: /tmp/test_kv.db
            sql_default:
              type: sql_sqlite
              db_path: /tmp/test_sql.db
          stores:
            metadata:
              backend: kv_default
              namespace: metadata
            inference:
              backend: sql_default
              table_name: inference
            conversations:
              backend: sql_default
              table_name: conversations
            responses:
              backend: sql_default
              table_name: responses
        providers:
          inference:
            - provider_id: provider1
              provider_type: remote::ollama
              config: {{}}
          safety:
            - provider_id: provider1
              provider_type: inline::llama-guard
              config:
                llama_guard_shield:
                  model: Llama-Guard-3-1B
                  excluded_categories: []
                  disable_input_check: false
                  disable_output_check: false
                enable_prompt_guard: false
          memory:
            - provider_id: provider1
              provider_type: inline::builtin
              config: {{}}
    """
    )


@pytest.fixture
def old_config():
    return yaml.safe_load(
        f"""
        distro_name: foo
        built_at: {datetime.now().isoformat()}
        apis_to_serve: []
        routing_table:
          inference:
            - provider_type: remote::ollama
              config:
                host: localhost
                port: 11434
              routing_key: Llama3.2-1B-Instruct
            - provider_type: remote::openai
              config:
                api_key: sk-test
              routing_key: Llama3.1-8B-Instruct
          safety:
            - routing_key: ["shield1", "shield2"]
              provider_type: inline::llama-guard
              config:
                llama_guard_shield:
                  model: Llama-Guard-3-1B
                  excluded_categories: []
                  disable_input_check: false
                  disable_output_check: false
                enable_prompt_guard: false
          memory:
            - routing_key: vector
              provider_type: inline::builtin
              config: {{}}
        api_providers:
    """
    )


@pytest.fixture
def invalid_config():
    return yaml.safe_load(
        """
        routing_table: {}
        api_providers: {}
    """
    )


def test_parse_and_maybe_upgrade_config_up_to_date(up_to_date_config):
    result = parse_and_maybe_upgrade_config(up_to_date_config)
    assert result.version == OGX_RUN_CONFIG_VERSION
    assert "inference" in result.providers


def test_parse_and_maybe_upgrade_config_old_format(old_config):
    result = parse_and_maybe_upgrade_config(old_config)
    assert result.version == OGX_RUN_CONFIG_VERSION
    assert all(api in result.providers for api in ["inference", "safety", "memory"])
    safety_provider = result.providers["safety"][0]
    assert safety_provider.provider_type == "inline::llama-guard"
    assert "llama_guard_shield" in safety_provider.config

    inference_providers = result.providers["inference"]
    assert len(inference_providers) == 2
    assert {x.provider_id for x in inference_providers} == {
        "remote::ollama-00",
        "remote::openai-01",
    }

    ollama = inference_providers[0]
    assert ollama.provider_type == "remote::ollama"
    assert ollama.config["port"] == 11434


def test_parse_and_maybe_upgrade_config_invalid(invalid_config):
    with pytest.raises(KeyError):
        parse_and_maybe_upgrade_config(invalid_config)


def test_parse_and_maybe_upgrade_config_distro_name_int(config_with_distro_name_int):
    result = parse_and_maybe_upgrade_config(config_with_distro_name_int)
    assert isinstance(result.distro_name, str)


def test_parse_and_maybe_upgrade_config_sets_external_providers_dir(up_to_date_config):
    """Test that external_providers_dir is None when not specified (deprecated field)."""
    # Ensure the config doesn't have external_providers_dir set
    assert "external_providers_dir" not in up_to_date_config

    result = parse_and_maybe_upgrade_config(up_to_date_config)

    # Verify external_providers_dir is None (not set to default)
    # This aligns with the deprecation of external_providers_dir
    assert result.external_providers_dir is None


def test_parse_and_maybe_upgrade_config_preserves_custom_external_providers_dir(up_to_date_config):
    """Test that custom external_providers_dir values are preserved."""
    custom_dir = "/custom/providers/dir"
    up_to_date_config["external_providers_dir"] = custom_dir

    result = parse_and_maybe_upgrade_config(up_to_date_config)

    # Verify the custom value was preserved
    assert str(result.external_providers_dir) == custom_dir


def test_generate_run_config_from_providers():
    """Test that run_config_from_dynamic_config_spec creates a valid config for the providers-run distro"""
    from ogx.core.stack import run_config_from_dynamic_config_spec

    config = run_config_from_dynamic_config_spec(
        "inference=remote::openai",
        distro_name="providers-run",
    )
    config_dict = config.model_dump(mode="json")

    # Verify basic structure
    assert config_dict["distro_name"] == "providers-run"
    assert "inference" in config_dict["apis"]
    assert "inference" in config_dict["providers"]

    # Verify storage has all required stores including prompts
    assert "storage" in config_dict
    stores = config_dict["storage"]["stores"]
    assert "prompts" in stores
    assert stores["prompts"]["namespace"] == "prompts"

    # Verify config can be parsed back
    parsed = parse_and_maybe_upgrade_config(config_dict)
    assert parsed.distro_name == "providers-run"


def test_providers_flag_generates_config_with_api_keys():
    """Test that --providers flag properly generates provider configs including API keys.

    This tests the fix where sample_run_config() is called to populate
    API keys and other credentials for remote providers like remote::openai.
    """
    import argparse
    from unittest.mock import patch

    from ogx.cli.stack.run import StackRun

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    stack_run = StackRun(subparsers)

    # Create args with --providers flag set
    args = argparse.Namespace(
        providers="inference=remote::openai",
        config=None,
        port=8321,
        image_type=None,
        distro_name=None,
        enable_ui=False,
    )

    # Mock _uvicorn_run to prevent starting a server
    with patch.object(stack_run, "_uvicorn_run"):
        stack_run._run_stack_run_cmd(args)

    # Read the generated config file
    from ogx.core.utils.config_dirs import DISTRIBS_BASE_DIR

    config_file = DISTRIBS_BASE_DIR / "providers-run" / "config.yaml"
    with open(config_file) as f:
        config_dict = yaml.safe_load(f)

    # Verify the provider has config with API keys
    inference_providers = config_dict["providers"]["inference"]
    assert len(inference_providers) == 1

    openai_provider = inference_providers[0]
    assert openai_provider["provider_type"] == "remote::openai"
    assert openai_provider["config"], "Provider config should not be empty"
    assert "api_key" in openai_provider["config"], "API key should be in provider config"
    assert "base_url" in openai_provider["config"], "Base URL should be in provider config"


@pytest.fixture
def config_with_image_name():
    """Test config using deprecated image_name field."""
    return yaml.safe_load(
        f"""
        version: {OGX_RUN_CONFIG_VERSION}
        image_name: my-old-distro
        apis_to_serve: []
        built_at: {datetime.now().isoformat()}
        storage:
          backends:
            kv_default:
              type: kv_sqlite
              db_path: /tmp/test_kv.db
            sql_default:
              type: sql_sqlite
              db_path: /tmp/test_sql.db
          stores:
            metadata:
              backend: kv_default
              namespace: metadata
            inference:
              backend: sql_default
              table_name: inference
            conversations:
              backend: sql_default
              table_name: conversations
            responses:
              backend: sql_default
              table_name: responses
            prompts:
              backend: kv_default
              namespace: prompts
        providers:
          inference:
            - provider_id: provider1
              provider_type: remote::ollama
              config: {{}}
    """
    )


def test_parse_config_with_deprecated_image_name(config_with_image_name):
    """Test that deprecated image_name field is properly migrated to distro_name."""
    import warnings

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = parse_and_maybe_upgrade_config(config_with_image_name)

        # Check that at least one deprecation warning about image_name was raised
        deprecation_warnings = [warning for warning in w if issubclass(warning.category, DeprecationWarning)]
        image_name_warnings = [
            warning
            for warning in deprecation_warnings
            if "image_name" in str(warning.message) and "distro_name" in str(warning.message)
        ]
        assert len(image_name_warnings) >= 1, "Expected at least one deprecation warning about image_name"

    # Verify distro_name is set from image_name
    assert result.distro_name == "my-old-distro"
    assert result.version == OGX_RUN_CONFIG_VERSION


def test_parse_config_with_both_names_prefers_distro_name():
    """Test that when both image_name and distro_name are provided, distro_name is used."""
    import warnings

    config = yaml.safe_load(
        f"""
        version: {OGX_RUN_CONFIG_VERSION}
        image_name: old-name
        distro_name: new-name
        apis_to_serve: []
        built_at: {datetime.now().isoformat()}
        storage:
          backends:
            kv_default:
              type: kv_sqlite
              db_path: /tmp/test_kv.db
            sql_default:
              type: sql_sqlite
              db_path: /tmp/test_sql.db
          stores:
            metadata:
              backend: kv_default
              namespace: metadata
            inference:
              backend: sql_default
              table_name: inference
            conversations:
              backend: sql_default
              table_name: conversations
            responses:
              backend: sql_default
              table_name: responses
            prompts:
              backend: kv_default
              namespace: prompts
        providers:
          inference:
            - provider_id: provider1
              provider_type: remote::ollama
              config: {{}}
    """
    )

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = parse_and_maybe_upgrade_config(config)

        # Check that deprecation warning was raised about both being provided
        deprecation_warnings = [warning for warning in w if issubclass(warning.category, DeprecationWarning)]
        both_provided_warnings = [warning for warning in deprecation_warnings if "Both" in str(warning.message)]
        assert len(both_provided_warnings) >= 1, (
            "Expected at least one deprecation warning about both fields being provided"
        )

    # Verify distro_name is preferred over image_name
    assert result.distro_name == "new-name"
