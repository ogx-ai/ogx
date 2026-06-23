# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

from ogx.core.datatypes import AccessRule, Api

from .config import LocalContainerRuntimeConfig

__all__ = ["LocalContainerRuntimeConfig", "LocalContainerRuntimeImpl"]


async def get_provider_impl(config: LocalContainerRuntimeConfig, deps: dict[Api, Any], policy: list[AccessRule]):
    from .impl import LocalContainerRuntimeImpl

    impl = LocalContainerRuntimeImpl(config, policy)
    await impl.initialize()
    return impl
