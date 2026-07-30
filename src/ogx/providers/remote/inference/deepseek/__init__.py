# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from .config import DeepSeekImplConfig


async def get_adapter_impl(config: DeepSeekImplConfig, _deps):
    from .deepseek import DeepSeekInferenceAdapter

    assert isinstance(config, DeepSeekImplConfig), f"Unexpected config type: {type(config)}"

    impl = DeepSeekInferenceAdapter(config=config)

    await impl.initialize()

    return impl
