# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

from ogx_api import Api

from .config import PlaceholderWebSearchConfig


async def get_provider_impl(config: PlaceholderWebSearchConfig, deps: dict[Api, Any]):
    from .placeholder import PlaceholderWebSearchImpl

    impl = PlaceholderWebSearchImpl(config)
    await impl.initialize()
    return impl
