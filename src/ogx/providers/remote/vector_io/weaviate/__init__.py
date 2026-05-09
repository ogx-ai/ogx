# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from ogx_api import Api, ProviderSpec

from .config import WeaviateVectorIOConfig


async def get_adapter_impl(config: WeaviateVectorIOConfig, deps: dict[Api, ProviderSpec]):
    from .weaviate import WeaviateVectorIOAdapter

    impl = WeaviateVectorIOAdapter(config, deps[Api.inference], deps.get(Api.files), deps.get(Api.file_processors))
    await impl.initialize()
    return impl
