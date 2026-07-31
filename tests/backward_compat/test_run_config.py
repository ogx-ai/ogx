# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""
Backward compatibility test for config.yaml files.

This test ensures that changes to StackRunConfig don't break
existing config.yaml files from previous versions.
"""

import os
from pathlib import Path

import pytest
import yaml

from ogx.core.datatypes import StackConfig
from ogx.core.distribution import builtin_automatically_routed_apis, get_provider_registry
from ogx.core.resolver import validate_and_prepare_providers
from ogx.core.stack import replace_env_vars


def get_test_configs():
    configs_dir = os.environ.get("COMPAT_TEST_CONFIGS_DIR")
    if configs_dir:
        # CI mode: test configs extracted from main/release
        config_dir = Path(configs_dir)
        if not config_dir.exists():
            pytest.skip(f"Config directory not found: {configs_dir}")

        config_files = sorted(config_dir.glob("*.yaml"))
        if not config_files:
            pytest.skip(f"No .yaml files found in {configs_dir}")

        return config_files
    else:
        # Local mode: test current distribution configs
        repo_root = Path(__file__).parent.parent.parent
        config_files = sorted((repo_root / "src" / "ogx" / "distributions").glob("*/config.yaml"))

        if not config_files:
            pytest.skip("No config.yaml files found in distributions/")

        return config_files


@pytest.mark.parametrize("config_file", get_test_configs(), ids=lambda p: p.parent.name)
def test_load_run_config(config_file):
    with open(config_file) as f:
        config_data = yaml.safe_load(f)

    # Resolve env vars for fields like discriminated unions (e.g. auth provider type)
    # that cannot accept raw ${env...} strings. Use replace_env_vars with fallback
    # to raw config for distributions requiring env vars without defaults.
    try:
        config_data = replace_env_vars(config_data)
    except Exception:
        pass

    # Validate schema
    config = StackConfig.model_validate(config_data)

    # Validate providers using the same validation the stack uses at runtime
    # This validates:
    #   - Provider types exist in registry
    #   - No conflicts with auto-routed APIs
    #   - Provider deprecation warnings/errors
    provider_registry = get_provider_registry()
    routing_table_apis = {x.routing_table_api for x in builtin_automatically_routed_apis()}
    router_apis = {x.router_api for x in builtin_automatically_routed_apis()}

    try:
        validate_and_prepare_providers(config, provider_registry, routing_table_apis, router_apis)
    except ValueError as e:
        pytest.fail(f"Provider validation failed: {str(e)}")
