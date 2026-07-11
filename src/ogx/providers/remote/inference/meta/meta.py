# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx

from ogx.log import get_logger
from ogx.providers.remote.inference.meta.config import MetaConfig
from ogx.providers.utils.inference.anthropic_translation import parse_anthropic_sse_event
from ogx.providers.utils.inference.http_client import (
    build_network_client_kwargs as _build_network_client_kwargs,
)
from ogx.providers.utils.inference.openai_mixin import OpenAIMixin
from ogx_api.messages.models import (
    ANTHROPIC_VERSION,
    AnthropicCountTokensRequest,
    AnthropicCountTokensResponse,
    AnthropicCreateMessageRequest,
    AnthropicMessageResponse,
    AnthropicStreamEvent,
)

logger = get_logger(name=__name__, category="inference::meta")


class MetaInferenceAdapter(OpenAIMixin):
    """Inference adapter for the Meta AI OpenAI-compatible API endpoint (api.meta.ai).

    Chat Completions and the Responses API are served by the OpenAI-compatible
    mixin. The endpoint also exposes the Anthropic Messages API natively, so
    ``anthropic_messages``/``anthropic_count_tokens`` forward directly to
    ``/v1/messages`` instead of using the mixin's translation fallback.
    """

    config: MetaConfig

    provider_data_api_key_field: str = "meta_api_key"

    def get_base_url(self) -> str:
        """Return the Meta AI API base URL (including the /v1 suffix)."""
        return str(self.config.base_url)

    def _get_messages_base_url(self) -> str:
        """Return the base URL without a trailing /v1 for building /v1/messages paths."""
        base_url = self.get_base_url().rstrip("/")
        if base_url.endswith("/v1"):
            base_url = base_url[:-3]
        return base_url

    def _build_httpx_client_kwargs(self) -> dict:
        """Build httpx.AsyncClient kwargs that honour network/TLS configuration."""
        kwargs = _build_network_client_kwargs(self.config.network)
        if not kwargs:
            kwargs["verify"] = self.shared_ssl_context
        return kwargs

    async def _passthrough_anthropic_messages(
        self,
        request: AnthropicCreateMessageRequest,
    ) -> AnthropicMessageResponse | AsyncIterator[AnthropicStreamEvent]:
        """Forward the request directly to Meta's /v1/messages endpoint."""
        url = f"{self._get_messages_base_url()}/v1/messages"
        body = request.model_dump(exclude_none=True)
        body["model"] = request.model
        headers = {
            "content-type": "application/json",
            "anthropic-version": ANTHROPIC_VERSION,
            "x-api-key": self._get_api_key_from_config_or_provider_data() or "no-key-required",
        }

        if request.stream:
            return self._passthrough_anthropic_stream(url, headers, body, timeout=300.0)

        async with httpx.AsyncClient(timeout=httpx.Timeout(300.0), **self._build_httpx_client_kwargs()) as client:
            resp = await client.post(url, json=body, headers=headers)
            resp.raise_for_status()
            return AnthropicMessageResponse(**resp.json())

    async def _passthrough_anthropic_stream(
        self,
        url: str,
        headers: dict[str, str],
        body: dict[str, Any],
        timeout: float = 300.0,
    ) -> AsyncIterator[AnthropicStreamEvent]:
        """Stream SSE events directly from Meta."""
        async with httpx.AsyncClient(timeout=timeout, **self._build_httpx_client_kwargs()) as client:
            async with client.stream("POST", url, json=body, headers=headers) as resp:
                resp.raise_for_status()
                event_type: str | None = None
                async for line in resp.aiter_lines():
                    line = line.strip()
                    if line.startswith("event: "):
                        event_type = line[7:]
                    elif line.startswith("data: ") and event_type:
                        data = json.loads(line[6:])
                        event = parse_anthropic_sse_event(event_type, data)
                        if event:
                            yield event
                        event_type = None

    async def anthropic_messages(
        self,
        params: AnthropicCreateMessageRequest,
    ) -> AnthropicMessageResponse | AsyncIterator[AnthropicStreamEvent]:
        """Handle Anthropic Messages via Meta's native /v1/messages endpoint."""
        return await self._passthrough_anthropic_messages(params)

    async def anthropic_count_tokens(
        self,
        params: AnthropicCountTokensRequest,
    ) -> AnthropicCountTokensResponse:
        """Forward count_tokens to Meta's /v1/messages/count_tokens endpoint."""
        url = f"{self._get_messages_base_url()}/v1/messages/count_tokens"
        body = params.model_dump(exclude_none=True)
        body["model"] = params.model
        headers = {
            "content-type": "application/json",
            "anthropic-version": ANTHROPIC_VERSION,
            "x-api-key": self._get_api_key_from_config_or_provider_data() or "no-key-required",
        }

        async with httpx.AsyncClient(timeout=httpx.Timeout(30.0), **self._build_httpx_client_kwargs()) as client:
            resp = await client.post(url, json=body, headers=headers)
            resp.raise_for_status()
            return AnthropicCountTokensResponse(**resp.json())
