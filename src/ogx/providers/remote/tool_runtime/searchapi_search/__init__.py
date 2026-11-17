# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from pydantic import BaseModel, SecretStr

from .config import SearchApiSearchToolConfig
from .searchapi_search import SearchApiSearchToolRuntimeImpl


class SearchApiSearchToolProviderDataValidator(BaseModel):
    """Validator for SearchApi Search tool provider data requiring a SearchApi API key."""

    searchapi_search_api_key: SecretStr


async def get_adapter_impl(config: SearchApiSearchToolConfig, _deps):
    impl = SearchApiSearchToolRuntimeImpl(config)
    await impl.initialize()
    return impl
