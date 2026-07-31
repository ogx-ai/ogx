# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from .config import SambaNovaImplConfig


async def get_adapter_impl(config: SambaNovaImplConfig, _deps):
    from .sambanova import SambaNovaInferenceAdapter

    assert isinstance(config, SambaNovaImplConfig), f"Unexpected config type: {type(config)}"
    impl = SambaNovaInferenceAdapter(config=config)
    await impl.initialize()
    return impl
