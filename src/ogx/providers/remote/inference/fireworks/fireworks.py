# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


from collections.abc import AsyncIterator
from typing import Any

from ogx.log import get_logger
from ogx.providers.utils.inference.openai_mixin import OpenAIMixin
from ogx_api import (
    OpenAIChatCompletion,
    OpenAIChatCompletionChunk,
    OpenAIChatCompletionRequestWithExtraBody,
    OpenAICompletion,
    OpenAICompletionRequestWithExtraBody,
)

from .config import FireworksImplConfig

logger = get_logger(name=__name__, category="inference::fireworks")


def _wants_usage(stream_options: dict[str, Any] | None) -> bool:
    return stream_options is not None and stream_options.get("include_usage") is True


def _strip_function_type_from_tools(tools: list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
    if tools is None:
        return None
    sanitized: list[dict[str, Any]] = []
    for tool in tools:
        function = tool.get("function") if isinstance(tool, dict) else None
        if isinstance(function, dict) and "type" in function:
            tool = {**tool, "function": {k: v for k, v in function.items() if k != "type"}}
        sanitized.append(tool)
    return sanitized


class FireworksInferenceAdapter(OpenAIMixin):
    """Inference adapter for the Fireworks AI platform."""

    config: FireworksImplConfig

    embedding_model_metadata: dict[str, dict[str, int]] = {
        "nomic-ai/nomic-embed-text-v1.5": {"embedding_dimension": 768, "context_length": 8192},
        "accounts/fireworks/models/qwen3-embedding-8b": {"embedding_dimension": 4096, "context_length": 40960},
    }

    provider_data_api_key_field: str = "fireworks_api_key"

    def get_base_url(self) -> str:
        return str(self.config.base_url)

    async def openai_chat_completion(
        self,
        params: OpenAIChatCompletionRequestWithExtraBody,
    ) -> OpenAIChatCompletion | AsyncIterator[OpenAIChatCompletionChunk]:
        # Fireworks rejects extra fields, including the "type" key inside tool
        # function definitions that some upstream converters emit.
        if params.tools:
            params = params.model_copy(update={"tools": _strip_function_type_from_tools(params.tools)})
        return await super().openai_chat_completion(params)

    async def openai_completion(
        self,
        params: OpenAICompletionRequestWithExtraBody,
    ) -> OpenAICompletion | AsyncIterator[OpenAICompletion]:
        # Fireworks appends a usage-only chunk (empty choices) to completions
        # streams unless include_usage is explicitly false. Opt out explicitly
        # when the client did not request usage.
        if params.stream and not _wants_usage(params.stream_options):
            params = params.model_copy(
                update={"stream_options": {**(params.stream_options or {}), "include_usage": False}}
            )
        return await super().openai_completion(params)
