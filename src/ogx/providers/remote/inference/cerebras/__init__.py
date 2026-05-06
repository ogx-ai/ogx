# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from .config import CerebrasImplConfig


async def get_adapter_impl(config: CerebrasImplConfig, _deps):
    from .cerebras import CerebrasInferenceAdapter

    assert isinstance(config, CerebrasImplConfig), f"Unexpected config type: {type(config)}"

    impl = CerebrasInferenceAdapter(config=config)

    await impl.initialize()

    return impl
