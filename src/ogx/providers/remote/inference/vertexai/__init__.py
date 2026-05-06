# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from .config import VertexAIConfig


async def get_adapter_impl(config: VertexAIConfig, _deps):
    from .vertexai import VertexAIInferenceAdapter

    impl = VertexAIInferenceAdapter(config=config)
    await impl.initialize()
    return impl
