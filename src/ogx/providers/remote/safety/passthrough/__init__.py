# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

from .config import PassthroughSafetyConfig


async def get_adapter_impl(config: PassthroughSafetyConfig, _deps: Any) -> Any:
    from .passthrough import PassthroughSafetyAdapter

    impl = PassthroughSafetyAdapter(config)
    await impl.initialize()
    return impl
