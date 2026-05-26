# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

from ogx.core.access_control.datatypes import AccessRule
from ogx_api import Api

from .config import SQLiteVectorIOConfig


async def get_provider_impl(config: SQLiteVectorIOConfig, deps: dict[Api, Any], policy: list[AccessRule] | None = None):
    from .sqlite_vec import SQLiteVecVectorIOAdapter

    assert isinstance(config, SQLiteVectorIOConfig), f"Unexpected config type: {type(config)}"
    impl = SQLiteVecVectorIOAdapter(
        config, deps[Api.inference], deps.get(Api.files), deps.get(Api.file_processors), policy=policy or []
    )
    await impl.initialize()
    return impl
