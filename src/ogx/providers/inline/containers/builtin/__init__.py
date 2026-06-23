# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

from ogx.core.datatypes import AccessRule, Api

from .config import BuiltinContainersConfig

__all__ = ["BuiltinContainersConfig", "BuiltinContainersImpl"]


async def get_provider_impl(config: BuiltinContainersConfig, deps: dict[Api, Any], policy: list[AccessRule]):
    from .impl import BuiltinContainersImpl

    impl = BuiltinContainersImpl(config, deps[Api.container_runtime], deps[Api.files], policy)
    await impl.initialize()
    return impl
