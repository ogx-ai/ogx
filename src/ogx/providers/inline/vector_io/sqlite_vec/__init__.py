# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

from ogx_api import Api

from .config import SQLiteVectorIOConfig


async def get_provider_impl(config: SQLiteVectorIOConfig, deps: dict[Api, Any]):
    from .sqlite_vec import SQLiteVecVectorIOAdapter

    assert isinstance(config, SQLiteVectorIOConfig), f"Unexpected config type: {type(config)}"
    impl = SQLiteVecVectorIOAdapter(config, deps[Api.inference], deps.get(Api.files), deps.get(Api.file_processors))
    await impl.initialize()
    return impl
