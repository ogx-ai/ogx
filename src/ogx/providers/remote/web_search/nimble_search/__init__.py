# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

from pydantic import BaseModel, SecretStr

from ogx_api import Api

from .config import NimbleSearchConfig
from .nimble_search import NimbleSearchWebSearchImpl


class NimbleSearchProviderDataValidator(BaseModel):
    """Validator for Nimble Search provider data requiring a Nimble API key."""

    nimble_search_api_key: SecretStr


async def get_adapter_impl(config: NimbleSearchConfig, _deps: dict[Api, Any]):
    impl = NimbleSearchWebSearchImpl(config)
    await impl.initialize()
    return impl
