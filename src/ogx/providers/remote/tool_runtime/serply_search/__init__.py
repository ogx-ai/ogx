# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from pydantic import BaseModel, SecretStr

from .config import SerplySearchToolConfig
from .serply_search import SerplySearchToolRuntimeImpl


class SerplySearchToolProviderDataValidator(BaseModel):
    """Validator for Serply Search tool provider data requiring a Serply API key."""

    serply_search_api_key: SecretStr


async def get_adapter_impl(config: SerplySearchToolConfig, _deps):
    impl = SerplySearchToolRuntimeImpl(config)
    await impl.initialize()
    return impl
