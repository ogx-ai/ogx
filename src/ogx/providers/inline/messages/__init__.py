# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

from ogx.core.access_control.datatypes import RouteAccessRule
from ogx.core.datatypes import AccessRule, Api
from ogx.core.storage.kvstore import kvstore_impl
from ogx.core.storage.sqlstore.authorized_sqlstore import authorized_sqlstore

from .config import MessagesConfig


async def get_provider_impl(
    config: MessagesConfig,
    deps: dict[Api, Any],
    policy: list[AccessRule],
    route_policy: list[RouteAccessRule],
):
    from .impl import BuiltinMessagesImpl

    try:
        sql_store = await authorized_sqlstore(config.sqlstore, policy)
    except ValueError as e:
        raise ValueError(
            f"Failed to initialize the messages provider: {e} Define that backend "
            "under storage.backends, or set sqlstore explicitly in the messages provider config."
        ) from e
    kvstore = await kvstore_impl(config.kvstore)
    impl = BuiltinMessagesImpl(config, deps[Api.inference], kvstore, sql_store, route_policy)
    await impl.initialize()
    return impl
