# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

from pydantic import BaseModel, SecretStr

from ogx_api import Api

from .config import ExaSearchConfig
from .exa_search import ExaSearchWebSearchImpl


class ExaSearchProviderDataValidator(BaseModel):
    """Validator for Exa Search provider data requiring an Exa API key."""

    exa_search_api_key: SecretStr


async def get_adapter_impl(config: ExaSearchConfig, _deps: dict[Api, Any]):
    impl = ExaSearchWebSearchImpl(config)
    await impl.initialize()
    return impl
