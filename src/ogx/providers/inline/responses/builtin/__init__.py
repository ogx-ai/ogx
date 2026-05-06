# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

from ogx.core.datatypes import AccessRule, Api

from .config import BuiltinResponsesImplConfig


async def get_provider_impl(
    config: BuiltinResponsesImplConfig,
    deps: dict[Api, Any],
    policy: list[AccessRule],
):
    from .impl import BuiltinResponsesImpl

    impl = BuiltinResponsesImpl(
        config,
        deps[Api.inference],
        deps[Api.vector_io],
        deps.get(Api.safety),
        deps[Api.tool_runtime],
        deps[Api.tool_groups],
        deps[Api.conversations],
        deps[Api.prompts],
        deps[Api.files],
        deps[Api.connectors],
        policy,
    )
    await impl.initialize()
    return impl
