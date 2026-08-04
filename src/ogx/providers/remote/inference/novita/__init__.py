# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from .config import NovitaImplConfig


async def get_adapter_impl(config: NovitaImplConfig, _deps):
    from .novita import NovitaInferenceAdapter

    assert isinstance(config, NovitaImplConfig), f"Unexpected config type: {type(config)}"

    impl = NovitaInferenceAdapter(config=config)

    await impl.initialize()

    return impl
