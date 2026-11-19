# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

from pydantic import BaseModel, SecretStr

from ogx_api import Api

from .brave_search import BraveSearchWebSearchImpl
from .config import BraveSearchConfig


class BraveSearchProviderDataValidator(BaseModel):
    """Validator for Brave Search provider data requiring a Brave Search API key."""

    brave_search_api_key: SecretStr


async def get_adapter_impl(config: BraveSearchConfig, _deps: dict[Api, Any]):
    impl = BraveSearchWebSearchImpl(config)
    await impl.initialize()
    return impl
