# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from pydantic import BaseModel, SecretStr

from .config import TogetherImplConfig


class TogetherProviderDataValidator(BaseModel):
    """Validator for Together provider data requiring a Together API key."""

    together_api_key: SecretStr


async def get_adapter_impl(config: TogetherImplConfig, _deps):
    from .together import TogetherInferenceAdapter

    assert isinstance(config, TogetherImplConfig), f"Unexpected config type: {type(config)}"
    impl = TogetherInferenceAdapter(config=config)
    await impl.initialize()
    return impl
