# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from .config import MCPProviderConfig


async def get_adapter_impl(config: MCPProviderConfig, _deps):
    from .model_context_protocol import ModelContextProtocolToolRuntimeImpl

    impl = ModelContextProtocolToolRuntimeImpl(config, _deps)
    await impl.initialize()
    return impl
