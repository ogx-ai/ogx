# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

from ogx_api import Api

from .config import MilvusVectorIOConfig


async def get_provider_impl(config: MilvusVectorIOConfig, deps: dict[Api, Any]):
    from ogx.providers.remote.vector_io.milvus.milvus import MilvusVectorIOAdapter

    impl = MilvusVectorIOAdapter(config, deps[Api.inference], deps.get(Api.files), deps.get(Api.file_processors))
    await impl.initialize()
    return impl
