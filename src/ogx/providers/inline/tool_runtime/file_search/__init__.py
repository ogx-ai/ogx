# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

from ogx_api import Api

from .config import FileSearchToolRuntimeConfig


async def get_provider_impl(config: FileSearchToolRuntimeConfig, deps: dict[Api, Any]):
    from .file_search import FileSearchToolRuntimeImpl

    impl = FileSearchToolRuntimeImpl(config, deps[Api.vector_io], deps[Api.inference], deps[Api.files])
    await impl.initialize()
    return impl
