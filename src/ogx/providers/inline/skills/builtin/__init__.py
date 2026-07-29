# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

from ogx.core.datatypes import Api
from ogx.core.storage.kvstore import kvstore_impl

from .config import BuiltinSkillsConfig


async def get_provider_impl(
    config: BuiltinSkillsConfig,
    deps: dict[Api, Any],
):
    from .impl import BuiltinSkillsImpl

    kvstore = await kvstore_impl(config.persistence)
    impl = BuiltinSkillsImpl(config, deps[Api.files], kvstore)
    await impl.initialize()
    return impl
