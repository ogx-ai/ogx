# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

from pydantic import BaseModel, SecretStr

from ogx_api import Api

from .config import TavilySearchConfig
from .tavily_search import TavilySearchWebSearchImpl


class TavilySearchProviderDataValidator(BaseModel):
    """Validator for Tavily Search provider data requiring a Tavily Search API key."""

    tavily_search_api_key: SecretStr


async def get_adapter_impl(config: TavilySearchConfig, _deps: dict[Api, Any]):
    impl = TavilySearchWebSearchImpl(config)
    await impl.initialize()
    return impl
