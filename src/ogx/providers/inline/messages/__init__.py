# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

from ogx.core.datatypes import Api
from ogx.core.storage.kvstore import kvstore_impl

from .config import MessagesConfig


async def get_provider_impl(
    config: MessagesConfig,
    deps: dict[Api, Any],
):
    from .impl import BuiltinMessagesImpl

    kvstore = await kvstore_impl(config.kvstore)
    impl = BuiltinMessagesImpl(config, deps[Api.inference], kvstore)
    await impl.initialize()
    return impl
