# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from collections.abc import AsyncIterator, Iterable
from typing import Any

import httpx
from anthropic import AsyncAnthropic

from ogx.providers.utils.inference.anthropic_translation import passthrough_anthropic_stream
from ogx.providers.utils.inference.http_client import build_network_client_kwargs
from ogx.providers.utils.inference.openai_mixin import OpenAIMixin
from ogx_api.inference.models import (
    OpenAIChatCompletion,
    OpenAIChatCompletionChunk,
    OpenAIChatCompletionRequestWithExtraBody,
    OpenAICompletion,
    OpenAICompletionRequestWithExtraBody,
)
from ogx_api.messages.models import (
    ANTHROPIC_VERSION,
    AnthropicCountTokensRequest,
    AnthropicCountTokensResponse,
    AnthropicCreateMessageRequest,
    AnthropicMessageResponse,
    AnthropicStreamEvent,
)

from .config import AnthropicConfig


def _make_schema_strict(schema: dict) -> None:
    """Recursively add additionalProperties: false to all object schemas for strict mode compliance."""
    if schema.get("type") == "object":
        if "additionalProperties" not in schema:
            schema["additionalProperties"] = False
        for prop in (schema.get("properties") or {}).values():
            _make_schema_strict(prop)


class AnthropicAPIError(Exception):
    """An error response from the Anthropic API, carrying its HTTP status so the server keeps it."""

    def __init__(self, status_code: int, message: str):
        super().__init__(message)
        self.status_code = status_code


def _error_message(response: httpx.Response) -> str:
    """The message from an Anthropic error body ({"type": "error", "error": {"message": ...}})."""
    try:
        body = response.json()
        message = body["error"]["message"]
        if isinstance(message, str):
            return message
    except (ValueError, KeyError, TypeError):
        pass
    return f"Anthropic API request failed with status {response.status_code}"


class AnthropicInferenceAdapter(OpenAIMixin):
    """Inference adapter for Anthropic Claude models.

    Chat Completions go through the OpenAI-compatible mixin. The Messages API is Anthropic's
    own wire format, so ``anthropic_messages``/``anthropic_count_tokens`` forward directly to
    ``/v1/messages`` instead of using the mixin's translation fallback, which cannot represent
    extended thinking.
    """

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

    def _build_httpx_client_kwargs(self, default_timeout: float) -> dict[str, Any]:
        """httpx client kwargs that honour config.network, else the shared SSL context.

        ``default_timeout`` applies only when ``network.timeout`` is unset.
        """
        kwargs = build_network_client_kwargs(self.config.network)
        if not kwargs:
            kwargs["verify"] = self.shared_ssl_context
        kwargs.setdefault("timeout", httpx.Timeout(default_timeout))
        return kwargs

    def _messages_headers(self) -> dict[str, str]:
        api_key = self._get_api_key_from_config_or_provider_data()
        if not api_key:
            raise ValueError(
                "API key not provided. Please provide a valid API key in the provider data header, "
                f'e.g. x-ogx-provider-data: {{"{self.provider_data_api_key_field}": "<API_KEY>"}}.'
            )
        return {
            "content-type": "application/json",
            "anthropic-version": ANTHROPIC_VERSION,
            "x-api-key": api_key,
        }

    async def _post(self, path: str, body: dict[str, Any], default_timeout: float) -> dict[str, Any]:
        url = f"{self.get_base_url().rstrip('/')}/{path}"
        headers = self._messages_headers()
        async with httpx.AsyncClient(**self._build_httpx_client_kwargs(default_timeout)) as client:
            resp = await client.post(url, json=body, headers=headers)
            if resp.is_error:
                raise AnthropicAPIError(resp.status_code, _error_message(resp))
            result: dict[str, Any] = resp.json()
            return result

    async def _stream_messages(
        self, url: str, headers: dict[str, str], body: dict[str, Any]
    ) -> AsyncIterator[AnthropicStreamEvent]:
        try:
            async for event in passthrough_anthropic_stream(
                url=url,
                req_body=body,
                headers=headers,
                httpx_client_kwargs=build_network_client_kwargs(self.config.network)
                or {"verify": self.shared_ssl_context},
            ):
                yield event
        except httpx.HTTPStatusError as e:
            # The response body is already closed here, so only the status is available.
            raise AnthropicAPIError(
                e.response.status_code,
                f"Anthropic API request failed with status {e.response.status_code}",
            ) from e

    async def anthropic_messages(
        self,
        params: AnthropicCreateMessageRequest,
    ) -> AnthropicMessageResponse | AsyncIterator[AnthropicStreamEvent]:
        """Forward the request to Anthropic's native /v1/messages endpoint."""
        self._validate_model_allowed(params.model)
        body = params.model_dump(exclude_none=True)
        if params.stream:
            # Resolve the headers here, not in the generator, so a missing API key fails the
            # request before any streaming starts.
            url = f"{self.get_base_url().rstrip('/')}/messages"
            return self._stream_messages(url, self._messages_headers(), body)
        return AnthropicMessageResponse(**await self._post("messages", body, default_timeout=300.0))

    async def anthropic_count_tokens(
        self,
        params: AnthropicCountTokensRequest,
    ) -> AnthropicCountTokensResponse:
        """Forward count_tokens to Anthropic's native /v1/messages/count_tokens endpoint."""
        self._validate_model_allowed(params.model)
        body = params.model_dump(exclude_none=True)
        return AnthropicCountTokensResponse(**await self._post("messages/count_tokens", body, default_timeout=30.0))

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
        if (
            params.response_format
            and hasattr(params.response_format, "json_schema")
            and params.response_format.json_schema
        ):
            js = params.response_format.json_schema
            # Anthropic requires strict: true for json_schema response format
            if js.get("strict") is None:
                js["strict"] = True
            schema = js.get("schema")
            if js["strict"] and isinstance(schema, dict):
                _make_schema_strict(schema)
        return await super().openai_chat_completion(params)

    async def openai_completion(
        self,
        params: OpenAICompletionRequestWithExtraBody,
    ) -> OpenAICompletion | AsyncIterator[OpenAICompletion]:
        """Anthropic does not support the /v1/completions endpoint."""
        raise NotImplementedError(
            "Anthropic does not support /v1/completions endpoint. Only /v1/chat/completions is supported. "
        )
