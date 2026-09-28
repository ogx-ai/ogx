# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from collections.abc import AsyncIterator

import httpx

from ogx.providers.utils.inference.anthropic_translation import passthrough_anthropic_stream
from ogx.providers.utils.inference.openai_mixin import OpenAIMixin
from ogx_api import (
    OpenAIChatCompletion,
    OpenAIChatCompletionChunk,
    OpenAIChatCompletionRequestWithExtraBody,
    OpenAICompletion,
    OpenAICompletionRequestWithExtraBody,
    OpenAIEmbeddingsRequestWithExtraBody,
    OpenAIEmbeddingsResponse,
)
from ogx_api.messages.models import (
    ANTHROPIC_VERSION,
    AnthropicCreateMessageRequest,
    AnthropicMessageResponse,
    AnthropicStreamEvent,
)

from .config import DeepSeekImplConfig


class DeepSeekInferenceAdapter(OpenAIMixin):
    """Inference adapter for the DeepSeek platform.

    DeepSeek exposes an OpenAI-compatible chat completions API, so the shared
    `OpenAIMixin` handles requests once pointed at DeepSeek's base URL. See
    https://api-docs.deepseek.com/.

    DeepSeek also exposes the Anthropic Messages API natively at a separate host path
    (config.anthropic_base_url, not a suffix of the OpenAI-compatible base_url), so
    ``anthropic_messages`` forwards directly to ``/v1/messages`` there instead of the mixin's
    translation fallback, which cannot represent extended thinking, cache control, or
    output_config.effort. DeepSeek has no /v1/messages/count_tokens endpoint, so
    ``anthropic_count_tokens`` is not overridden: the mixin's default implementation counts
    tokens by calling ``anthropic_messages`` (this override) with ``max_tokens=1``, the same
    fallback every other provider without a native counting endpoint uses.
    """

    config: DeepSeekImplConfig

    provider_data_api_key_field: str = "deepseek_api_key"

    def get_base_url(self) -> str:
        return str(self.config.base_url)

    async def anthropic_messages(
        self,
        params: AnthropicCreateMessageRequest,
    ) -> AnthropicMessageResponse | AsyncIterator[AnthropicStreamEvent]:
        """Forward the request to DeepSeek's native /v1/messages endpoint."""
        url = f"{str(self.config.anthropic_base_url).rstrip('/')}/v1/messages"
        body = params.model_dump(exclude_none=True)
        body["model"] = params.model
        headers = {
            "content-type": "application/json",
            "anthropic-version": ANTHROPIC_VERSION,
            "x-api-key": self._get_api_key_from_config_or_provider_data() or "no-key-required",
        }

        if params.stream:
            return passthrough_anthropic_stream(
                url=url,
                req_body=body,
                headers=headers,
                httpx_client_kwargs=self._build_httpx_client_kwargs(),
            )

        async with httpx.AsyncClient(**self._build_httpx_client_kwargs(default_timeout=300.0)) as client:
            resp = await client.post(url, json=body, headers=headers)
            resp.raise_for_status()
            return AnthropicMessageResponse(**resp.json())

    async def openai_chat_completion(
        self,
        params: OpenAIChatCompletionRequestWithExtraBody,
    ) -> OpenAIChatCompletion | AsyncIterator[OpenAIChatCompletionChunk]:
        if params.response_format is not None and params.response_format.type == "json_schema":
            raise ValueError(
                "DeepSeek does not support response_format type 'json_schema'. Use 'json_object' or 'text' instead."
            )
        return await super().openai_chat_completion(params)

    async def openai_embeddings(
        self,
        params: OpenAIEmbeddingsRequestWithExtraBody,
    ) -> OpenAIEmbeddingsResponse:
        raise NotImplementedError("DeepSeek does not expose an embeddings endpoint.")

    async def openai_completion(
        self,
        params: OpenAICompletionRequestWithExtraBody,
    ) -> OpenAICompletion | AsyncIterator[OpenAICompletion]:
        """DeepSeek does not support the legacy /v1/completions endpoint.

        DeepSeek's completion API exists only as a beta FIM feature behind a
        separate base URL (https://api.deepseek.com/beta), so it is not
        reachable through this adapter's standard API surface.
        """
        raise NotImplementedError(
            "DeepSeek does not support /v1/completions endpoint. Only /v1/chat/completions is supported."
        )
