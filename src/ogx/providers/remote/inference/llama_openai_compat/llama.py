# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from collections.abc import AsyncIterator

from ogx.log import get_logger
from ogx.providers.remote.inference.llama_openai_compat.config import LlamaCompatConfig
from ogx.providers.utils.inference.openai_mixin import OpenAIMixin
from ogx_api import (
    OpenAICompletion,
    OpenAICompletionRequestWithExtraBody,
    OpenAIEmbeddingsRequestWithExtraBody,
    OpenAIEmbeddingsResponse,
)

logger = get_logger(name=__name__, category="inference::llama_openai_compat")


class LlamaCompatInferenceAdapter(OpenAIMixin):
    """Inference adapter for Llama models using an OpenAI-compatible API endpoint."""

    config: LlamaCompatConfig

    provider_data_api_key_field: str = "llama_api_key"
    """
    Llama API Inference Adapter for OGX.
    """

    def get_base_url(self) -> str:
        """
        Get the base URL for OpenAI mixin.

        :return: The Llama API base URL
        """
        return str(self.config.base_url)

    async def openai_completion(
        self,
        params: OpenAICompletionRequestWithExtraBody,
    ) -> OpenAICompletion | AsyncIterator[OpenAICompletion]:
        raise NotImplementedError()

    async def openai_embeddings(
        self,
        params: OpenAIEmbeddingsRequestWithExtraBody,
    ) -> OpenAIEmbeddingsResponse:
        raise NotImplementedError()
