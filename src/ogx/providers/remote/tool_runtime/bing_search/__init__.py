# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from .bing_search import BingSearchToolRuntimeImpl
from .config import BingSearchToolConfig

__all__ = ["BingSearchToolConfig", "BingSearchToolRuntimeImpl"]
from pydantic import BaseModel, SecretStr


class BingSearchToolProviderDataValidator(BaseModel):
    bing_search_api_key: SecretStr


async def get_adapter_impl(config: BingSearchToolConfig, _deps):
    impl = BingSearchToolRuntimeImpl(config)
    await impl.initialize()
    return impl
