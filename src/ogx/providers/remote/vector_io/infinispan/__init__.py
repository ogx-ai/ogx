# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from ogx.core.access_control.datatypes import AccessRule
from ogx_api import Api, ProviderSpec

from .config import InfinispanVectorIOConfig


async def get_adapter_impl(
    config: InfinispanVectorIOConfig, deps: dict[Api, ProviderSpec], policy: list[AccessRule] | None = None
):
    from .infinispan import InfinispanVectorIOAdapter

    impl = InfinispanVectorIOAdapter(
        config,
        deps[Api.inference],  # type: ignore[arg-type]
        deps.get(Api.files),  # type: ignore[arg-type]
        deps.get(Api.file_processors),  # type: ignore[arg-type]
        policy=policy or [],
    )
    await impl.initialize()
    return impl
