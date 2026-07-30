# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import os
from pathlib import Path

import pytest
import yaml
from openai import OpenAI

from ogx.core.library_client import OGXAsLibraryClient

# --- Helper Functions ---


def _load_all_verification_configs():
    """Load and aggregate verification configs from the conf/ directory."""
    # Note: Path is relative to *this* file (fixtures.py)
    conf_dir = Path(__file__).parent.parent.parent / "conf"
    if not conf_dir.is_dir():
        # Use pytest.fail if called during test collection, otherwise raise error
        # For simplicity here, we'll raise an error, assuming direct calls
        # are less likely or can handle it.
        raise FileNotFoundError(f"Verification config directory not found at {conf_dir}")

    all_provider_configs = {}
    yaml_files = list(conf_dir.glob("*.yaml"))
    if not yaml_files:
        raise FileNotFoundError(f"No YAML configuration files found in {conf_dir}")

    for config_path in yaml_files:
        provider_name = config_path.stem
        try:
            with open(config_path) as f:
                provider_config = yaml.safe_load(f)
                if provider_config:
                    all_provider_configs[provider_name] = provider_config
                else:
                    # Log warning if possible, or just skip empty files silently
                    print(f"Warning: Config file {config_path} is empty or invalid.")
        except Exception as e:
            raise OSError(f"Error loading config file {config_path}: {e}") from e

    return {"providers": all_provider_configs}


# --- End Helper Functions ---


@pytest.fixture(scope="session")
def verification_config():
    """Pytest fixture to provide the loaded verification config."""
    try:
        return _load_all_verification_configs()
    except (OSError, FileNotFoundError) as e:
        pytest.fail(str(e))  # Fail test collection if config loading fails


@pytest.fixture(scope="session")
def provider(request, verification_config):
    provider = request.config.getoption("--provider")
    base_url = request.config.getoption("--base-url")

    if provider and base_url and verification_config["providers"][provider]["base_url"] != base_url:
        raise ValueError(f"Provider {provider} is not supported for base URL {base_url}")

    if not provider:
        if not base_url:
            raise ValueError("Provider and base URL are not provided")
        for provider, metadata in verification_config["providers"].items():
            if metadata["base_url"] == base_url:
                provider = provider
                break

    return provider


@pytest.fixture(scope="session")
def base_url(request, provider, verification_config):
    return request.config.getoption("--base-url") or verification_config.get("providers", {}).get(provider, {}).get(
        "base_url"
    )


@pytest.fixture(scope="session")
def api_key(request, provider, verification_config):
    provider_conf = verification_config.get("providers", {}).get(provider, {})
    api_key_env_var = provider_conf.get("api_key_var")

    key_from_option = request.config.getoption("--api-key")
    key_from_env = os.getenv(api_key_env_var) if api_key_env_var else None

    final_key = key_from_option or key_from_env
    return final_key


@pytest.fixture
def model_mapping(provider, providers_model_mapping):
    return providers_model_mapping[provider]


@pytest.fixture(scope="session")
def openai_client(base_url, api_key, provider):
    # Simplify running against a local OGX
    if base_url and "localhost" in base_url and not api_key:
        api_key = "empty"
    if provider.startswith("stack:"):
        parts = provider.split(":")
        if len(parts) != 2:
            raise ValueError(f"Invalid config for OGX: {provider}, it must be of the form 'stack:<config>'")
        config = parts[1]
        client = OGXAsLibraryClient(config, skip_logger_removal=True)
        return client

    client = OpenAI(
        base_url=base_url,
        api_key=api_key,
        max_retries=0,
        timeout=60.0,
    )
    yield client
    # Cleanup: close HTTP connections
    try:
        client.close()
    except Exception:
        pass
