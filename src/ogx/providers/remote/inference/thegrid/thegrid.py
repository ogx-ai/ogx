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

from .config import TheGridImplConfig


class TheGridInferenceAdapter(OpenAIMixin):
    """Inference adapter for The Grid.

    The Grid is a spot market for inference: model ids are market instruments
    such as `text-standard`, `code-prime` or `agent-max` rather than fixed
    models, and a request is filled by whichever supplier is competitive at the
    time. Because of this, the `model` field of a response names the model that
    actually served the request, which will differ from the instrument that was
    requested.

    The chat completions API is OpenAI-compatible, so the shared `OpenAIMixin`
    handles requests once pointed at The Grid's base URL, and the instrument
    list is discovered from `/v1/models`. See https://thegrid.ai/docs.
    """

    config: TheGridImplConfig

    provider_data_api_key_field: str = "thegrid_api_key"

    def get_base_url(self) -> str:
        return str(self.config.base_url)

    async def openai_embeddings(
        self,
        params: OpenAIEmbeddingsRequestWithExtraBody,
    ) -> OpenAIEmbeddingsResponse:
        raise NotImplementedError("The Grid does not expose an embeddings endpoint.")

    async def openai_completion(
        self,
        params: OpenAICompletionRequestWithExtraBody,
    ) -> OpenAICompletion | AsyncIterator[OpenAICompletion]:
        """The Grid does not serve the legacy /v1/completions endpoint.

        Only /v1/chat/completions is available; the legacy route returns 404.
        """
        raise NotImplementedError(
            "The Grid does not support /v1/completions endpoint. Only /v1/chat/completions is supported."
        )
