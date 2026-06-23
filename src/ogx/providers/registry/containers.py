# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from ogx_api import Api, InlineProviderSpec, ProviderSpec


def available_providers() -> list[ProviderSpec]:
    """Return the list of available containers provider specifications."""
    return [
        InlineProviderSpec(
            api=Api.containers,
            provider_type="inline::builtin",
            pip_packages=[],
            module="ogx.providers.inline.containers.builtin",
            config_class="ogx.providers.inline.containers.builtin.config.BuiltinContainersConfig",
            api_dependencies=[
                Api.container_runtime,
                Api.files,
            ],
            description=(
                "Serves the Containers API (POST/GET/DELETE /v1alpha/containers and file "
                "sub-resources). Enforces operator network-policy layering, seeds requested "
                "file_ids from the Files API, and delegates all lifecycle, file, and "
                "shell-execution work to a configured container_runtime provider."
            ),
        ),
    ]
