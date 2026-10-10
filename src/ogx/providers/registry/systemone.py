# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from ogx_api import Api, ProviderSpec, RemoteProviderSpec


def available_providers() -> list[ProviderSpec]:
    return [
        RemoteProviderSpec(
            api=Api.systemone,
            adapter_type="typesafe",
            provider_type="remote::typesafe",
            pip_packages=[],
            module="ogx.providers.remote.systemone.typesafe",
            config_class="ogx.providers.remote.systemone.typesafe.config.TypeSafeConfig",
            provider_data_validator="ogx.providers.remote.systemone.typesafe.config.TypeSafeProviderDataValidator",
            description="TypeSafe (Jev) decision model provider, the canonical implementation of the SystemOne judgment wire format (POST /v1/systemone).",
        ),
    ]
