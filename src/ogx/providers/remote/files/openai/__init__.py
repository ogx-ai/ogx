# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

from ogx.core.datatypes import AccessRule, Api

from .config import OpenAIFilesImplConfig


async def get_adapter_impl(config: OpenAIFilesImplConfig, deps: dict[Api, Any], policy: list[AccessRule] | None = None):
    from .files import OpenAIFilesImpl

    impl = OpenAIFilesImpl(config, policy or [])
    await impl.initialize()
    return impl
