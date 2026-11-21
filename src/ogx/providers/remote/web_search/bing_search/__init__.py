# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

from pydantic import BaseModel, SecretStr

from ogx_api import Api

from .bing_search import BingSearchWebSearchImpl
from .config import BingSearchConfig


class BingSearchProviderDataValidator(BaseModel):
    """Validator for Bing Search provider data requiring a Bing Search API key."""

    bing_search_api_key: SecretStr


async def get_adapter_impl(config: BingSearchConfig, _deps: dict[Api, Any]):
    impl = BingSearchWebSearchImpl(config)
    await impl.initialize()
    return impl
