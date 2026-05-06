# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from pydantic import BaseModel, SecretStr

from .config import FireworksImplConfig


class FireworksProviderDataValidator(BaseModel):
    """Validator for Fireworks provider data requiring a Fireworks API key."""

    fireworks_api_key: SecretStr


async def get_adapter_impl(config: FireworksImplConfig, _deps):
    from .fireworks import FireworksInferenceAdapter

    assert isinstance(config, FireworksImplConfig), f"Unexpected config type: {type(config)}"
    impl = FireworksInferenceAdapter(config=config)
    await impl.initialize()
    return impl
