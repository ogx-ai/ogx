# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from ogx.core.storage.sqlstore.sqlstore import sql_store_pip_packages
from ogx_api import Api, InlineProviderSpec, ProviderSpec


def available_providers() -> list[ProviderSpec]:
    """Return the list of available container_runtime provider specifications.

    The container_runtime API backs the public Containers API: providers
    implement the backend lifecycle (Docker/Podman, Kubernetes) that the
    Containers service delegates to.
    """
    return [
        InlineProviderSpec(
            api=Api.container_runtime,
            provider_type="inline::local",
            pip_packages=["docker"] + sql_store_pip_packages,
            module="ogx.providers.inline.container_runtime.local",
            config_class="ogx.providers.inline.container_runtime.local.config.LocalContainerRuntimeConfig",
            description=(
                "Local container runtime backed by Docker or Podman via the docker-py SDK. "
                "Manages sandbox lifecycle, file operations, shell execution, and skill "
                "mounting against a local container engine socket. Container and file "
                "metadata is persisted in a SQL store; expired containers are reaped by a "
                "background task."
            ),
        ),
    ]
