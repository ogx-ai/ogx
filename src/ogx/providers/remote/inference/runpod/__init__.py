# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from .config import RunpodImplConfig


async def get_adapter_impl(config: RunpodImplConfig, _deps):
    from .runpod import RunpodInferenceAdapter

    assert isinstance(config, RunpodImplConfig), f"Unexpected config type: {type(config)}"
    impl = RunpodInferenceAdapter(config=config)
    await impl.initialize()
    return impl
