# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from collections.abc import AsyncIterator

from ogx.providers.utils.inference.anthropic_mixin import AnthropicMixin
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

from .config import DeepSeekImplConfig


class DeepSeekInferenceAdapter(AnthropicMixin, OpenAIMixin):
    """Inference adapter for the DeepSeek platform.

    DeepSeek exposes an OpenAI-compatible chat completions API, so the shared
    `OpenAIMixin` handles requests once pointed at DeepSeek's base URL. See
    https://api-docs.deepseek.com/.

    DeepSeek also exposes the Anthropic Messages API natively at a separate host path
    (config.anthropic_base_url, not a suffix of the OpenAI-compatible base_url), so
    ``AnthropicMixin`` forwards directly to ``/v1/messages`` there instead of the
    translation fallback, which cannot represent extended thinking, cache control, or
    output_config.effort. DeepSeek has no /v1/messages/count_tokens endpoint, so
    ``_anthropic_count_tokens_url`` returns None and counting falls through to
    ``OpenAIMixin``'s default, which counts by calling ``anthropic_messages`` with
    ``max_tokens=1``, the same fallback every other provider without a native counting
    endpoint uses.
    """

    config: DeepSeekImplConfig

    provider_data_api_key_field: str = "deepseek_api_key"

    def get_base_url(self) -> str:
        return str(self.config.base_url)

    def _get_anthropic_base_url(self) -> str:
        # DeepSeek's Anthropic surface is a separate host path, not derived from base_url.
        return str(self.config.anthropic_base_url).rstrip("/")

    def _anthropic_count_tokens_url(self) -> str | None:
        """DeepSeek has no native /v1/messages/count_tokens endpoint."""
        return None

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
