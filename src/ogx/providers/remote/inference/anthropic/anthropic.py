# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from collections.abc import AsyncIterator, Iterable

from anthropic import AsyncAnthropic

from ogx.providers.utils.inference.openai_mixin import OpenAIMixin
from ogx_api.inference.models import (
    OpenAIChatCompletion,
    OpenAIChatCompletionChunk,
    OpenAIChatCompletionRequestWithExtraBody,
    OpenAICompletion,
    OpenAICompletionRequestWithExtraBody,
)

from .config import AnthropicConfig


class AnthropicInferenceAdapter(OpenAIMixin):
    """Inference adapter for Anthropic Claude models."""

    config: AnthropicConfig

    provider_data_api_key_field: str = "anthropic_api_key"
    # source: https://docs.claude.com/en/docs/build-with-claude/embeddings
    # TODO: add support for voyageai, which is where these models are hosted
    # embedding_model_metadata = {
    #     "voyage-3-large": {"embedding_dimension": 1024, "context_length": 32000},  # supports dimensions 256, 512, 1024, 2048
    #     "voyage-3.5": {"embedding_dimension": 1024, "context_length": 32000},  # supports dimensions 256, 512, 1024, 2048
    #     "voyage-3.5-lite": {"embedding_dimension": 1024, "context_length": 32000},  # supports dimensions 256, 512, 1024, 2048
    #     "voyage-code-3": {"embedding_dimension": 1024, "context_length": 32000},  # supports dimensions 256, 512, 1024, 2048
    #     "voyage-finance-2": {"embedding_dimension": 1024, "context_length": 32000},
    #     "voyage-law-2": {"embedding_dimension": 1024, "context_length": 16000},
    #     "voyage-multimodal-3": {"embedding_dimension": 1024, "context_length": 32000},
    # }

    def get_base_url(self):
        return "https://api.anthropic.com/v1"

    async def list_provider_model_ids(self) -> Iterable[str]:
        api_key = self._get_api_key_from_config_or_provider_data()
        return [m.id async for m in AsyncAnthropic(api_key=api_key).models.list()]

    async def openai_chat_completion(
        self,
        params: OpenAIChatCompletionRequestWithExtraBody,
    ) -> OpenAIChatCompletion | AsyncIterator[OpenAIChatCompletionChunk]:
        # Anthropic rejects parameters: {} but OpenAI accepts it
        if params.tools:
            for tool in params.tools:
                func = tool.get("function", {})
                p = func.get("parameters")
                if isinstance(p, dict) and not p:
                    func["parameters"] = {"type": "object"}
        return await super().openai_chat_completion(params)

    async def openai_completion(
        self,
        params: OpenAICompletionRequestWithExtraBody,
    ) -> OpenAICompletion | AsyncIterator[OpenAICompletion]:
        """Anthropic does not support the /v1/completions endpoint."""
        raise NotImplementedError(
            "Anthropic does not support /v1/completions endpoint. Only /v1/chat/completions is supported. "
        )
