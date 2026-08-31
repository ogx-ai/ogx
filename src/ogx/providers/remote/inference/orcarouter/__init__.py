# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from .config import OrcaRouterImplConfig


async def get_adapter_impl(config: OrcaRouterImplConfig, _deps):
    from .orcarouter import OrcaRouterInferenceAdapter

    assert isinstance(config, OrcaRouterImplConfig), f"Unexpected config type: {type(config)}"

    impl = OrcaRouterInferenceAdapter(config=config)

    await impl.initialize()

    return impl
