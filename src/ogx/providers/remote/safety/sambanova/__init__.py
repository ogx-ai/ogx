# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


from typing import Any

from .config import SambaNovaSafetyConfig


async def get_adapter_impl(config: SambaNovaSafetyConfig, _deps) -> Any:
    from .sambanova import SambaNovaSafetyAdapter

    impl = SambaNovaSafetyAdapter(config)
    await impl.initialize()
    return impl
