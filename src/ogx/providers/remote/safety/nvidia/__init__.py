# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


from typing import Any

from .config import NVIDIASafetyConfig


async def get_adapter_impl(config: NVIDIASafetyConfig, _deps) -> Any:
    from .nvidia import NVIDIASafetyAdapter

    impl = NVIDIASafetyAdapter(config)
    await impl.initialize()
    return impl
