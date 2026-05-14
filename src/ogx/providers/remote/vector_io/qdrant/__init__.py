# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from ogx.core.access_control.datatypes import AccessRule
from ogx_api import Api, ProviderSpec

from .config import QdrantVectorIOConfig


async def get_adapter_impl(
    config: QdrantVectorIOConfig, deps: dict[Api, ProviderSpec], policy: list[AccessRule] | None = None
):
    from .qdrant import QdrantVectorIOAdapter

    impl = QdrantVectorIOAdapter(
        config, deps[Api.inference], deps.get(Api.files), deps.get(Api.file_processors), policy=policy or []
    )
    await impl.initialize()
    return impl
