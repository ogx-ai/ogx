# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from ogx.core.storage.sqlstore.sqlstore import sql_store_pip_packages
from ogx_api import Api, InlineProviderSpec, ProviderSpec, RemoteProviderSpec


def available_providers() -> list[ProviderSpec]:
    """Return the list of available container provider specifications."""
    return [
        InlineProviderSpec(
            api=Api.containers,
            provider_type="inline::docker",
            pip_packages=["docker"] + sql_store_pip_packages,
            module="ogx.providers.inline.containers.docker",
            config_class="ogx.providers.inline.containers.docker.config.DockerContainersImplConfig",
            description="Docker-based container provider for managing sandboxed execution environments.",
        ),
        RemoteProviderSpec(
            api=Api.containers,
            provider_type="remote::openshell",
            adapter_type="openshell",
            pip_packages=["openshell", "grpcio"] + sql_store_pip_packages,
            module="ogx.providers.remote.containers.openshell",
            config_class="ogx.providers.remote.containers.openshell.config.OpenShellContainersImplConfig",
            description="OpenShell-based container provider for security-hardened sandboxed execution.",
        ),
    ]
