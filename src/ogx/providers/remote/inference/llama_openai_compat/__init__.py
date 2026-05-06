# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from .config import LlamaCompatConfig


async def get_adapter_impl(config: LlamaCompatConfig, _deps):
    # import dynamically so the import is used only when it is needed
    from .llama import LlamaCompatInferenceAdapter

    adapter = LlamaCompatInferenceAdapter(config=config)
    return adapter
