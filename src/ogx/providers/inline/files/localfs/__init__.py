# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

from ogx.core.datatypes import AccessRule, Api

from .config import LocalfsFilesImplConfig
from .files import LocalfsFilesImpl

__all__ = ["LocalfsFilesImpl", "LocalfsFilesImplConfig"]


async def get_provider_impl(config: LocalfsFilesImplConfig, deps: dict[Api, Any], policy: list[AccessRule]):
    impl = LocalfsFilesImpl(config, policy)
    await impl.initialize()
    return impl
