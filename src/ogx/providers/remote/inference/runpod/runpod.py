# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from ogx.providers.utils.inference.openai_mixin import OpenAIMixin

from .config import RunpodImplConfig


class RunpodInferenceAdapter(OpenAIMixin):
    """
    Adapter for RunPod's OpenAI-compatible API endpoints.
    Supports VLLM for serverless endpoint self-hosted or public endpoints.
    Can work with any runpod endpoints that support OpenAI-compatible API
    """

    config: RunpodImplConfig
    provider_data_api_key_field: str = "runpod_api_token"

    def get_base_url(self) -> str:
        """Get base URL for OpenAI client."""
        return str(self.config.base_url)
