# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

from ogx.core.datatypes import AccessRule, Api

from .config import DockerContainersImplConfig
from .containers import DockerContainersImpl

__all__ = ["DockerContainersImpl", "DockerContainersImplConfig"]


async def get_provider_impl(config: DockerContainersImplConfig, deps: dict[Api, Any], policy: list[AccessRule]):
    impl = DockerContainersImpl(config, policy)
    await impl.initialize()
    return impl
