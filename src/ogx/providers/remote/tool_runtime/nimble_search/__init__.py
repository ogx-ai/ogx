# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from pydantic import BaseModel, SecretStr

from .config import NimbleSearchToolConfig
from .nimble_search import NimbleSearchToolRuntimeImpl


class NimbleSearchToolProviderDataValidator(BaseModel):
    """Validator for Nimble Search tool provider data requiring a Nimble API key."""

    nimble_search_api_key: SecretStr


async def get_adapter_impl(config: NimbleSearchToolConfig, _deps):
    impl = NimbleSearchToolRuntimeImpl(config)
    await impl.initialize()
    return impl
