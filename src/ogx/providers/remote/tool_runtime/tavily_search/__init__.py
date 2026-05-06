# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from pydantic import BaseModel, SecretStr

from .config import TavilySearchToolConfig
from .tavily_search import TavilySearchToolRuntimeImpl


class TavilySearchToolProviderDataValidator(BaseModel):
    """Validator for Tavily Search tool provider data requiring a Tavily Search API key."""

    tavily_search_api_key: SecretStr


async def get_adapter_impl(config: TavilySearchToolConfig, _deps):
    impl = TavilySearchToolRuntimeImpl(config)
    await impl.initialize()
    return impl
