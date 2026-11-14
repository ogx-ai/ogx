# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from pydantic import BaseModel, SecretStr

from .config import SGLangInferenceAdapterConfig


class SGLangProviderDataValidator(BaseModel):
    """Validator for SGLang provider data with an optional API token."""

    sglang_api_token: SecretStr | None = None


async def get_adapter_impl(config: SGLangInferenceAdapterConfig, _deps):
    from .sglang import SGLangInferenceAdapter

    assert isinstance(config, SGLangInferenceAdapterConfig), f"Unexpected config type: {type(config)}"
    impl = SGLangInferenceAdapter(config=config)
    await impl.initialize()
    return impl
