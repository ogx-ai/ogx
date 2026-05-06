# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from ogx.cli.stack.utils import add_dependent_providers
from ogx.core.datatypes import Provider
from ogx.core.distribution import get_provider_registry


def test_add_dependent_providers_expands_required_apis():
    provider_registry = get_provider_registry()
    provider_list = {
        "agents": [
            Provider(
                provider_type="inline::builtin",
                provider_id="builtin",
            )
        ]
    }

    add_dependent_providers(
        provider_list=provider_list,
        provider_registry=provider_registry,
        requested_provider_types=["inline::builtin"],
    )

    # Required API dependencies for agents should be present.
    assert "inference" in provider_list
    assert "vector_io" in provider_list
    assert "tool_runtime" in provider_list
    assert "files" in provider_list

    # Providers should be added for those APIs.
    assert provider_list["inference"]
    assert provider_list["vector_io"]
    assert provider_list["tool_runtime"]
    assert provider_list["files"]


def test_add_dependent_providers_include_configs():
    provider_registry = get_provider_registry()
    provider_list = {
        "agents": [
            Provider(
                provider_type="inline::builtin",
                provider_id="builtin",
            )
        ]
    }

    add_dependent_providers(
        provider_list=provider_list,
        provider_registry=provider_registry,
        requested_provider_types=["inline::builtin"],
        include_configs=True,
        distro_dir="~/.llama/distributions/providers-run",
    )

    # Some providers like sentence-transformers don't need configuration,
    # so they may have empty configs. Check providers that have actual config needs.
    vector_io_provider = provider_list["vector_io"][0]
    assert vector_io_provider.config, "Expected sample config for vector_io provider"
    assert "persistence" in vector_io_provider.config

    files_provider = provider_list["files"][0]
    assert files_provider.config, "Expected sample config for files provider"
    assert "storage_dir" in files_provider.config
