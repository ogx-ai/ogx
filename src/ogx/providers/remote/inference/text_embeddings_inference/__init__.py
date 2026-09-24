# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from .config import TextEmbeddingsInferenceConfig


async def get_adapter_impl(config: TextEmbeddingsInferenceConfig, _deps):
    # import dynamically so the import is used only when it is needed
    from .text_embeddings_inference import TextEmbeddingsInferenceAdapter

    adapter = TextEmbeddingsInferenceAdapter(config=config)
    await adapter.initialize()
    return adapter
