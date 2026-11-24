# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

from pydantic import BaseModel, SecretStr

from ogx_api import Api

from .config import SearchApiSearchConfig
from .searchapi_search import SearchApiSearchWebSearchImpl


class SearchApiSearchProviderDataValidator(BaseModel):
    """Validator for SearchApi Search provider data requiring a SearchApi API key."""

    searchapi_search_api_key: SecretStr


async def get_adapter_impl(config: SearchApiSearchConfig, _deps: dict[Api, Any]):
    impl = SearchApiSearchWebSearchImpl(config)
    await impl.initialize()
    return impl
