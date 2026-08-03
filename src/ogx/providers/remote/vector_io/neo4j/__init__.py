# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import cast

from ogx.core.access_control.datatypes import AccessRule
from ogx_api import Api, FileProcessors, Files, Inference, ProviderSpec

from .config import Neo4jVectorIOConfig


async def get_adapter_impl(
    config: Neo4jVectorIOConfig,
    deps: dict[Api, ProviderSpec],
    policy: list[AccessRule] | None = None,
):
    from .neo4j import Neo4jVectorIOAdapter

    impl = Neo4jVectorIOAdapter(
        config,
        cast(Inference, deps[Api.inference]),
        cast(Files | None, deps.get(Api.files)),
        cast(FileProcessors | None, deps.get(Api.file_processors)),
        policy=policy or [],
    )
    await impl.initialize()
    return impl
