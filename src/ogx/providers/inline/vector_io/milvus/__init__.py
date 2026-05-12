# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

from ogx.core.access_control.datatypes import AccessRule
from ogx_api import Api

from .config import MilvusVectorIOConfig


async def get_provider_impl(config: MilvusVectorIOConfig, deps: dict[Api, Any], policy: list[AccessRule] | None = None):
    from ogx.providers.remote.vector_io.milvus.milvus import MilvusVectorIOAdapter

    impl = MilvusVectorIOAdapter(
        config, deps[Api.inference], deps.get(Api.files), deps.get(Api.file_processors), policy=policy or []
    )
    await impl.initialize()
    return impl
