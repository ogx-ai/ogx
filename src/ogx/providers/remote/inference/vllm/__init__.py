# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from pydantic import BaseModel, SecretStr

from .config import VLLMInferenceAdapterConfig


class VLLMProviderDataValidator(BaseModel):
    """Validator for vLLM provider data with an optional API token."""

    vllm_api_token: SecretStr | None = None


async def get_adapter_impl(config: VLLMInferenceAdapterConfig, _deps):
    from .vllm import VLLMInferenceAdapter

    assert isinstance(config, VLLMInferenceAdapterConfig), f"Unexpected config type: {type(config)}"
    impl = VLLMInferenceAdapter(config=config)
    await impl.initialize()
    return impl
