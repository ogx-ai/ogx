# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

from ogx_api import Api

from .config import FaissVectorIOConfig


async def get_provider_impl(config: FaissVectorIOConfig, deps: dict[Api, Any]):
    from .faiss import FaissVectorIOAdapter

    assert isinstance(config, FaissVectorIOConfig), f"Unexpected config type: {type(config)}"

    impl = FaissVectorIOAdapter(config, deps[Api.inference], deps.get(Api.files), deps.get(Api.file_processors))
    await impl.initialize()
    return impl
