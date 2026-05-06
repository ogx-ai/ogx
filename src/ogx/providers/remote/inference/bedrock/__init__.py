# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
from .config import BedrockConfig


async def get_adapter_impl(config: BedrockConfig, _deps):
    from .bedrock import BedrockInferenceAdapter

    if not isinstance(config, BedrockConfig):
        raise TypeError(f"Unexpected config type: {type(config)}")

    impl = BedrockInferenceAdapter(config=config)

    await impl.initialize()

    return impl
