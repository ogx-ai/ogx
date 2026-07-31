# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from pydantic import BaseModel, SecretStr

from .config import WolframAlphaToolConfig
from .wolfram_alpha import WolframAlphaToolRuntimeImpl

__all__ = ["WolframAlphaToolConfig", "WolframAlphaToolRuntimeImpl"]


class WolframAlphaToolProviderDataValidator(BaseModel):
    wolfram_alpha_api_key: SecretStr


async def get_adapter_impl(config: WolframAlphaToolConfig, _deps):
    impl = WolframAlphaToolRuntimeImpl(config)
    await impl.initialize()
    return impl
