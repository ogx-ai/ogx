# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from .config import AzureConfig


async def get_adapter_impl(config: AzureConfig, _deps):
    from .azure import AzureInferenceAdapter

    impl = AzureInferenceAdapter(config=config)
    await impl.initialize()
    return impl
