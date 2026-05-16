# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from ogx.providers.registry.utils import ProviderRegistry
from ogx_api import Api

from ..datatypes import InlineProviderSpec

CONTAINERS_PROVIDERS = ProviderRegistry(
    {
        "inline::docker": InlineProviderSpec(
            api=Api.containers,
            provider_type="inline::docker",
            module="ogx.providers.inline.containers.docker",
            config_class="ogx.providers.inline.containers.docker.config.DockerContainersImplConfig",
            api_dependencies=[],
        ),
    }
)
