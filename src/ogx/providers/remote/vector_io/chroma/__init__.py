# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from ogx_api import Api, ProviderSpec

from .config import ChromaVectorIOConfig


async def get_adapter_impl(config: ChromaVectorIOConfig, deps: dict[Api, ProviderSpec]):
    from .chroma import ChromaVectorIOAdapter

    impl = ChromaVectorIOAdapter(config, deps[Api.inference], deps.get(Api.files), deps.get(Api.file_processors))
    await impl.initialize()
    return impl
