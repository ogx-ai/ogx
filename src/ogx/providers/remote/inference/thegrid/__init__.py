# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from .config import TheGridImplConfig


async def get_adapter_impl(config: TheGridImplConfig, _deps):
    from .thegrid import TheGridInferenceAdapter

    assert isinstance(config, TheGridImplConfig), f"Unexpected config type: {type(config)}"

    impl = TheGridInferenceAdapter(config=config)

    await impl.initialize()

    return impl
