# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

from ogx_api import Api

from .config import PlaceholderWebSearchToolConfig


async def get_provider_impl(config: PlaceholderWebSearchToolConfig, deps: dict[Api, Any]):
    from .placeholder import PlaceholderWebSearchToolRuntimeImpl

    impl = PlaceholderWebSearchToolRuntimeImpl(config)
    await impl.initialize()
    return impl
