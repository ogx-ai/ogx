# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from pydantic import BaseModel, SecretStr

from .brave_search import BraveSearchToolRuntimeImpl
from .config import BraveSearchToolConfig


class BraveSearchToolProviderDataValidator(BaseModel):
    """Validator for Brave Search tool provider data requiring a Brave Search API key."""

    brave_search_api_key: SecretStr


async def get_adapter_impl(config: BraveSearchToolConfig, _deps):
    impl = BraveSearchToolRuntimeImpl(config)
    await impl.initialize()
    return impl
