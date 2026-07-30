# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from ogx.core.access_control.datatypes import AccessRule
from ogx_api import Api, ProviderSpec

from .config import MilvusVectorIOConfig


async def get_adapter_impl(
    config: MilvusVectorIOConfig, deps: dict[Api, ProviderSpec], policy: list[AccessRule] | None = None
):
    from .milvus import MilvusVectorIOAdapter

    assert isinstance(config, MilvusVectorIOConfig), f"Unexpected config type: {type(config)}"
    impl = MilvusVectorIOAdapter(
        config, deps[Api.inference], deps.get(Api.files), deps.get(Api.file_processors), policy=policy or []
    )
    await impl.initialize()
    return impl
