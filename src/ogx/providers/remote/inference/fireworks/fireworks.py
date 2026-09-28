# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


from collections.abc import AsyncIterator
from typing import Any

import httpx

from ogx.log import get_logger
from ogx.providers.utils.inference.anthropic_translation import passthrough_anthropic_stream
from ogx.providers.utils.inference.openai_mixin import OpenAIMixin
from ogx_api import (
    OpenAIChatCompletion,
    OpenAIChatCompletionChunk,
    OpenAIChatCompletionRequestWithExtraBody,
    OpenAICompletion,
    OpenAICompletionRequestWithExtraBody,
)
from ogx_api.messages.models import (
    ANTHROPIC_VERSION,
    AnthropicCreateMessageRequest,
    AnthropicMessageResponse,
    AnthropicStreamEvent,
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
    """Inference adapter for the Fireworks AI platform.

    Chat Completions go through the OpenAI-compatible mixin. Fireworks also exposes the
    Anthropic Messages API natively, so ``anthropic_messages`` forwards directly to
    ``/v1/messages`` instead of the mixin's translation fallback, which cannot represent
    extended thinking, cache control or tool search. Fireworks has no
    ``/v1/messages/count_tokens`` endpoint, so ``anthropic_count_tokens`` is not overridden:
    the mixin's default implementation counts tokens by calling ``anthropic_messages`` (this
    override) with ``max_tokens=1``, the same fallback every other provider without a native
    counting endpoint uses.
    """

    config: FireworksImplConfig

    embedding_model_metadata: dict[str, dict[str, int]] = {
        "nomic-ai/nomic-embed-text-v1.5": {"embedding_dimension": 768, "context_length": 8192},
        "accounts/fireworks/models/qwen3-embedding-8b": {"embedding_dimension": 4096, "context_length": 40960},
    }

    provider_data_api_key_field: str = "fireworks_api_key"

    def get_base_url(self) -> str:
        return str(self.config.base_url)

    def _get_messages_base_url(self) -> str:
        """Return the base URL without a trailing /v1, for building /v1/messages paths."""
        base_url = self.get_base_url().rstrip("/")
        if base_url.endswith("/v1"):
            base_url = base_url[:-3]
        return base_url

    async def anthropic_messages(
        self,
        params: AnthropicCreateMessageRequest,
    ) -> AnthropicMessageResponse | AsyncIterator[AnthropicStreamEvent]:
        """Forward the request to Fireworks' native /v1/messages endpoint."""
        url = f"{self._get_messages_base_url()}/v1/messages"
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
