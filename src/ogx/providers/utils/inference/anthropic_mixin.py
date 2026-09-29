# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from collections.abc import AsyncIterator
from typing import Any, ClassVar

import httpx

from ogx.providers.utils.inference.anthropic_translation import passthrough_anthropic_stream
from ogx.providers.utils.inference.openai_mixin import OpenAIMixin
from ogx_api.messages.models import (
    ANTHROPIC_VERSION,
    AnthropicCountTokensRequest,
    AnthropicCountTokensResponse,
    AnthropicCreateMessageRequest,
    AnthropicMessageResponse,
    AnthropicStreamEvent,
)


class AnthropicAPIError(Exception):
    """An error response from an Anthropic-compatible API, carrying its HTTP status so the server keeps it."""

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
    return f"Failed to complete Anthropic API request: status {response.status_code}"


class AnthropicMixin(OpenAIMixin):
    """Native Anthropic Messages API passthrough, shared by providers with a ``/v1/messages`` endpoint.

    Declare it before ``OpenAIMixin`` in the base list so it shadows the mixin's
    OpenAI-translation fallback, which cannot represent extended thinking, cache
    control or tool search::

        class FooInferenceAdapter(AnthropicMixin, OpenAIMixin): ...

    Providers configure it with ``ClassVar`` class attributes (so they stay class
    variables rather than pydantic fields):

    - ``anthropic_auth_style``: ``"x-api-key"`` (default) or ``"bearer"``.
    - ``anthropic_no_key_placeholder``: value sent as the auth header when no key
      is configured. ``None`` omits the header for the ``"bearer"`` style and
      raises ``ValueError`` (with a provider-data hint) for the ``"x-api-key"`` style.
    - ``anthropic_messages_timeout`` / ``anthropic_count_tokens_timeout``: default
      client timeouts in seconds; a configured ``network.timeout`` takes precedence.

    Providers without a ``/v1/messages/count_tokens`` endpoint override
    :meth:`_anthropic_count_tokens_url` to return ``None``; counting then falls through to
    ``OpenAIMixin``'s default, which counts by calling ``anthropic_messages`` with
    ``max_tokens=1``.

    By default the endpoint URL is built from ``get_base_url()`` (with a trailing
    ``/v1`` stripped). Providers whose Anthropic surface lives at a different host
    path than their OpenAI-compatible one (DeepSeek) override
    :meth:`_get_anthropic_base_url` instead.
    """

    anthropic_auth_style: ClassVar[str] = "x-api-key"
    anthropic_no_key_placeholder: ClassVar[str | None] = "no-key-required"
    anthropic_messages_timeout: ClassVar[float] = 300.0
    anthropic_count_tokens_timeout: ClassVar[float] = 30.0

    def _get_anthropic_base_url(self) -> str:
        """The provider base URL without a trailing ``/v1``, for building ``/v1/messages`` paths."""
        base_url = str(self.get_base_url()).rstrip("/")
        if base_url.endswith("/v1"):
            base_url = base_url[:-3]
        return base_url

    def _anthropic_count_tokens_url(self) -> str | None:
        """The native count_tokens URL, or ``None`` if the provider has no such endpoint."""
        return f"{self._get_anthropic_base_url()}/v1/messages/count_tokens"

    def _anthropic_auth_header(self) -> tuple[str, str] | None:
        api_key = self._get_api_key_from_config_or_provider_data()
        if api_key and api_key != "NO KEY REQUIRED":
            if self.anthropic_auth_style == "bearer":
                return ("Authorization", f"Bearer {api_key}")
            return ("x-api-key", api_key)
        if self.anthropic_auth_style == "bearer":
            return None
        if self.anthropic_no_key_placeholder is not None:
            return ("x-api-key", self.anthropic_no_key_placeholder)
        raise ValueError(
            "API key not provided. Please provide a valid API key in the provider data header, "
            f'e.g. x-ogx-provider-data: {{"{self.provider_data_api_key_field}": "<API_KEY>"}}.'
        )

    def _anthropic_headers(self) -> dict[str, str]:
        headers = {
            "content-type": "application/json",
            "anthropic-version": ANTHROPIC_VERSION,
        }
        auth = self._anthropic_auth_header()
        if auth is not None:
            headers[auth[0]] = auth[1]
        return headers

    async def _passthrough_anthropic_stream(
        self,
        url: str,
        headers: dict[str, str],
        req_body: dict[str, Any],
    ) -> AsyncIterator[AnthropicStreamEvent]:
        try:
            async for event in passthrough_anthropic_stream(
                url=url,
                req_body=req_body,
                headers=headers,
                httpx_client_kwargs=self._build_httpx_client_kwargs(self.anthropic_messages_timeout),
            ):
                yield event
        except httpx.HTTPStatusError as e:
            # The response body is already closed here, so only the status is available.
            raise AnthropicAPIError(
                e.response.status_code,
                f"Failed to complete Anthropic API request: status {e.response.status_code}",
            ) from e

    async def anthropic_messages(
        self,
        params: AnthropicCreateMessageRequest,
    ) -> AnthropicMessageResponse | AsyncIterator[AnthropicStreamEvent]:
        """Forward the request to the provider's native /v1/messages endpoint."""
        self._validate_model_allowed(params.model)
        url = f"{self._get_anthropic_base_url()}/v1/messages"
        body = params.model_dump(exclude_none=True)
        # Resolve the headers here, not in the generator, so a missing API key fails the
        # request before any streaming starts.
        headers = self._anthropic_headers()

        if params.stream:
            return self._passthrough_anthropic_stream(url, headers, body)

        async with httpx.AsyncClient(**self._build_httpx_client_kwargs(self.anthropic_messages_timeout)) as client:
            try:
                resp = await client.post(url, json=body, headers=headers)
                resp.raise_for_status()
            except httpx.HTTPStatusError as e:
                raise AnthropicAPIError(e.response.status_code, _error_message(e.response)) from e
            return AnthropicMessageResponse(**resp.json())

    async def anthropic_count_tokens(
        self,
        params: AnthropicCountTokensRequest,
    ) -> AnthropicCountTokensResponse:
        """Forward count_tokens to the provider's native /v1/messages/count_tokens endpoint."""
        self._validate_model_allowed(params.model)
        url = self._anthropic_count_tokens_url()
        if url is None:
            # No native endpoint: count via OpenAIMixin's default, which calls
            # anthropic_messages (this override) with max_tokens=1.
            return await super().anthropic_count_tokens(params)

        body = params.model_dump(exclude_none=True)
        headers = self._anthropic_headers()

        async with httpx.AsyncClient(**self._build_httpx_client_kwargs(self.anthropic_count_tokens_timeout)) as client:
            try:
                resp = await client.post(url, json=body, headers=headers)
                resp.raise_for_status()
            except httpx.HTTPStatusError as e:
                raise AnthropicAPIError(e.response.status_code, _error_message(e.response)) from e
            return AnthropicCountTokensResponse(**resp.json())
