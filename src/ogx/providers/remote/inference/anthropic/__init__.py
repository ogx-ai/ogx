# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from .config import AnthropicConfig


async def get_adapter_impl(config: AnthropicConfig, _deps):
    from .anthropic import AnthropicInferenceAdapter

    impl = AnthropicInferenceAdapter(config=config)
    await impl.initialize()
    return impl
