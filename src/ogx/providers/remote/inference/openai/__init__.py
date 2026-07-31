# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from .config import OpenAIConfig


async def get_adapter_impl(config: OpenAIConfig, _deps):
    from .openai import OpenAIInferenceAdapter

    impl = OpenAIInferenceAdapter(config=config)
    await impl.initialize()
    return impl
