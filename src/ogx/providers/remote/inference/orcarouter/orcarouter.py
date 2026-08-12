# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from collections.abc import AsyncIterator

from ogx.providers.remote.inference.orcarouter.config import OrcaRouterImplConfig
from ogx.providers.utils.inference.openai_mixin import OpenAIMixin
from ogx_api import OpenAICompletion, OpenAICompletionRequestWithExtraBody


class OrcaRouterInferenceAdapter(OpenAIMixin):
    """Inference adapter for the OrcaRouter gateway platform.

    OrcaRouter is an OpenAI-compatible aggregation gateway that routes requests
    to a range of frontier and open models through a single endpoint. The shared
    `OpenAIMixin` handles chat completions and embeddings once pointed at
    OrcaRouter's base URL. See https://www.orcarouter.ai.
    """

    config: OrcaRouterImplConfig

    provider_data_api_key_field: str = "orcarouter_api_key"

    def get_base_url(self) -> str:
        return str(self.config.base_url)

    async def openai_completion(
        self,
        params: OpenAICompletionRequestWithExtraBody,
    ) -> OpenAICompletion | AsyncIterator[OpenAICompletion]:
        """OrcaRouter does not support the legacy /v1/completions endpoint.

        OrcaRouter is a chat-first aggregation gateway, so the legacy
        OpenAI completions endpoint is not part of its API surface.
        """
        raise NotImplementedError(
            "OrcaRouter does not support /v1/completions endpoint. Only /v1/chat/completions is supported."
        )
