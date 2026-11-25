# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

from ogx.core.access_control.datatypes import AccessRule
from ogx.core.datatypes import Api
from ogx.core.storage.kvstore import kvstore_impl
from ogx.core.storage.sqlstore.authorized_sqlstore import authorized_sqlstore

from .config import BuiltinSkillsConfig


async def get_provider_impl(
    config: BuiltinSkillsConfig,
    deps: dict[Api, Any],
    policy: list[AccessRule],
):
    from .impl import BuiltinSkillsImpl

    try:
        sql_store = await authorized_sqlstore(config.sqlstore, policy)
    except ValueError as e:
        raise ValueError(
            f"Failed to initialize the skills provider: {e} Define that backend "
            "under storage.backends, or set sqlstore explicitly in the skills provider config."
        ) from e
    kvstore = await kvstore_impl(config.persistence)
    impl = BuiltinSkillsImpl(config, deps[Api.files], kvstore, sql_store)
    await impl.initialize()
    return impl
