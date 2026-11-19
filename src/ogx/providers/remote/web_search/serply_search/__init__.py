# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

from pydantic import BaseModel, SecretStr

from ogx_api import Api

from .config import SerplySearchConfig
from .serply_search import SerplySearchWebSearchImpl


class SerplySearchProviderDataValidator(BaseModel):
    """Validator for Serply Search provider data requiring a Serply API key."""

    serply_search_api_key: SecretStr


async def get_adapter_impl(config: SerplySearchConfig, _deps: dict[Api, Any]):
    impl = SerplySearchWebSearchImpl(config)
    await impl.initialize()
    return impl
