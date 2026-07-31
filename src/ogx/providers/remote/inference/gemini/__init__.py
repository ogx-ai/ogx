# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from .config import GeminiConfig


async def get_adapter_impl(config: GeminiConfig, _deps):
    from .gemini import GeminiInferenceAdapter

    impl = GeminiInferenceAdapter(config=config)
    await impl.initialize()
    return impl
