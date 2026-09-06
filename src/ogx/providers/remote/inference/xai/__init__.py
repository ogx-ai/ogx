# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from .config import XAIImplConfig


async def get_adapter_impl(config: XAIImplConfig, _deps):
    from .xai import XAIInferenceAdapter

    assert isinstance(config, XAIImplConfig), f"Unexpected config type: {type(config)}"

    impl = XAIInferenceAdapter(config=config)

    await impl.initialize()

    return impl
