# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from .config import MistralImplConfig


async def get_adapter_impl(config: MistralImplConfig, _deps):
    from .mistral import MistralInferenceAdapter

    assert isinstance(config, MistralImplConfig), f"Unexpected config type: {type(config)}"

    impl = MistralInferenceAdapter(config=config)

    await impl.initialize()

    return impl
