# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from collections.abc import AsyncIterator

from ogx.providers.utils.inference.openai_mixin import OpenAIMixin
from ogx_api import (
    OpenAICompletion,
    OpenAICompletionRequestWithExtraBody,
    OpenAIEmbeddingsRequestWithExtraBody,
    OpenAIEmbeddingsResponse,
)

from .config import MistralImplConfig


class MistralInferenceAdapter(OpenAIMixin):
    """Inference adapter for the Mistral AI platform.

    Mistral exposes an OpenAI-compatible API (chat completions and embeddings),
    so the shared `OpenAIMixin` handles requests once pointed at Mistral's base
    URL. See https://docs.mistral.ai/api/.
    """

    config: MistralImplConfig

    provider_data_api_key_field: str = "mistral_api_key"

    embedding_model_metadata: dict[str, dict[str, int]] = {
        "mistral-embed": {"embedding_dimension": 1024, "context_length": 8192},
    }

    def get_base_url(self) -> str:
        return str(self.config.base_url)

    async def openai_embeddings(
        self,
        params: OpenAIEmbeddingsRequestWithExtraBody,
    ) -> OpenAIEmbeddingsResponse:
        """Mistral's embeddings endpoint does not support encoding_format.

        Mistral accepts the param but ignores it, always returning a list of
        floats.  Rather than silently accepting a misleading parameter, we
        reject base64 requests explicitly so users get a clear error.
        """
        if params.encoding_format == "base64":
            raise ValueError("Mistral's embeddings endpoint does not support encoding_format='base64'.")

        return await super().openai_embeddings(params)

    async def openai_completion(
        self,
        params: OpenAICompletionRequestWithExtraBody,
    ) -> OpenAICompletion | AsyncIterator[OpenAICompletion]:
        """Mistral does not support the legacy /v1/completions endpoint."""
        raise NotImplementedError(
            "Mistral does not support /v1/completions endpoint. Only /v1/chat/completions is supported. "
        )
