# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from ogx.providers.utils.inference.openai_mixin import OpenAIMixin
from ogx_api import (
    OpenAIEmbeddingsRequestWithExtraBody,
    OpenAIEmbeddingsResponse,
)

from .config import DeepSeekImplConfig


class DeepSeekInferenceAdapter(OpenAIMixin):
    """Inference adapter for the DeepSeek platform.

    DeepSeek exposes an OpenAI-compatible chat completions API, so the shared
    `OpenAIMixin` handles requests once pointed at DeepSeek's base URL. See
    https://api-docs.deepseek.com/.
    """

    config: DeepSeekImplConfig

    provider_data_api_key_field: str = "deepseek_api_key"

    def get_base_url(self) -> str:
        return str(self.config.base_url)

    async def openai_embeddings(
        self,
        params: OpenAIEmbeddingsRequestWithExtraBody,
    ) -> OpenAIEmbeddingsResponse:
        # DeepSeek does not expose an embeddings endpoint.
        raise NotImplementedError()
