# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from .config import LlamaCppServerConfig


async def get_adapter_impl(config: LlamaCppServerConfig, _deps):
    # import dynamically so the import is used only when it is needed
    from .llama_cpp_server import LlamaCppServerInferenceAdapter

    adapter = LlamaCppServerInferenceAdapter(config=config)
    return adapter
