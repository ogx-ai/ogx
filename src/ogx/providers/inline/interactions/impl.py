# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Built-in Google Interactions API implementation.

Translates Google Interactions format to/from OpenAI Chat Completions format,
delegating to the inference API for actual model calls. When the underlying
inference provider natively supports the Google Interactions API (e.g. Gemini),
requests are forwarded directly without translation.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, TypedDict

import httpx

from ogx.core.access_control.access_control import is_action_allowed
from ogx.core.access_control.conditions import User as AccessControlUser
from ogx.core.access_control.datatypes import Action
from ogx.core.request_headers import get_authenticated_user
from ogx.log import get_logger
from ogx.providers.utils.inference.http_client import build_network_client_kwargs
from ogx.providers.utils.inference.model_registry import NetworkConfig
from ogx_api import (
    Inference,
    OpenAIChatCompletion,
    OpenAIChatCompletionChunk,
    OpenAIChatCompletionRequestWithExtraBody,
)
from ogx_api.interactions import Interactions
from ogx_api.interactions.models import (
    ContentDeltaEvent,
    ContentStartEvent,
    ContentStopEvent,
    GoogleCreateInteractionRequest,
    GoogleGenerationConfig,
    GoogleInputTurn,
    GoogleInteractionResponse,
    GoogleOutput,
    GoogleStreamEvent,
    GoogleTextOutput,
    GoogleThoughtOutput,
    GoogleUsage,
    InteractionCompleteEvent,
    InteractionStartEvent,
    _ContentRef,
    _InteractionCompleteRef,
    _InteractionRef,
    _TextDelta,
)

from .config import InteractionsConfig

logger = get_logger(name=__name__, category="interactions")


def _now_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S+00:00")


class _RawSSEStream(AsyncIterator[str]):
    """Async iterator that forwards raw SSE lines from an upstream provider.

    Marked with ``_raw_sse = True`` so the FastAPI route can stream it
    directly without re-serialisation.

    """

    _raw_sse = True

    def __init__(self, url: str, body: dict[str, Any], client_kwargs: dict[str, Any]):
        self._url = url
        self._body = body
        self._client_kwargs = client_kwargs
        self._iterator: AsyncIterator[str] | None = None

    def __aiter__(self) -> _RawSSEStream:
        return self

    async def __anext__(self) -> str:
        if self._iterator is None:
            self._iterator = self._stream()
        return await self._iterator.__anext__()

    async def _stream(self) -> AsyncIterator[str]:
        async with httpx.AsyncClient(**self._client_kwargs) as client:
            async with client.stream("POST", self._url, json=self._body) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    yield line + "\n"


class _PassthroughInfo(TypedDict):
    base_url: str
    auth_headers: dict[str, str]
    provider_resource_id: str
    network_config: NetworkConfig | None


@dataclass
class _FallbackModelResource:
    type: str
    identifier: str
    owner: AccessControlUser | None = None


class BuiltinInteractionsImpl(Interactions):
    """Google Interactions API adapter that translates to the inference API."""

    def __init__(self, config: InteractionsConfig, inference_api: Inference):
        self.config = config
        self.inference_api = inference_api

    async def initialize(self) -> None:
        pass

    async def shutdown(self) -> None:
        pass

    async def create_interaction(
        self,
        request: GoogleCreateInteractionRequest,
    ) -> GoogleInteractionResponse | AsyncIterator[GoogleStreamEvent] | AsyncIterator[str]:
        passthrough = await self._get_passthrough_info(request.model)
        if passthrough:
            return await self._passthrough_request(passthrough, request)

        openai_params = self._google_to_openai(request)

        result = await self.inference_api.openai_chat_completion(openai_params)

        if isinstance(result, AsyncIterator):
            return self._stream_openai_to_google(result, request.model)

        return self._openai_to_google(result, request.model)

    # -- Native passthrough for providers with /interactions support --

    # Module paths of provider impls known to support /interactions natively
    _NATIVE_INTERACTIONS_MODULES = {"ogx.providers.remote.inference.gemini"}

    async def _get_passthrough_info(self, model: str) -> _PassthroughInfo | None:
        """Check if the model's provider supports /interactions natively.

        Returns passthrough config for native /interactions, or None to use translation.
        """
        routing_table = getattr(self.inference_api, "routing_table", None)
        if routing_table is None:
            return None

        obj = await routing_table.get_object_by_identifier("model", model)
        provider_resource_id = obj.provider_resource_id if obj else None

        # Fall back to provider_id/model_id format (e.g. "gemini/gemini-2.5-flash")
        # to match the inference router's _get_provider_by_fallback behavior
        if obj is None:
            splits = model.split("/", maxsplit=1)
            if len(splits) != 2:
                return None
            provider_id, provider_resource_id = splits
            if provider_id not in routing_table.impls_by_provider_id:
                return None

            # Mirror inference fallback RBAC checks for provider_id/model_id lookups.
            temp_model = _FallbackModelResource(
                type="model",
                identifier=model,
            )
            user = get_authenticated_user()
            if not is_action_allowed(routing_table.policy, Action.READ, temp_model, user):
                logger.debug(
                    "Access denied to model via interactions fallback path",
                    model=model,
                    user=user.principal if user else "anonymous",
                )
                return None

            provider_impl = routing_table.impls_by_provider_id[provider_id]
        else:
            provider_impl = await routing_table.get_provider_impl(obj.identifier)

        provider_module = type(provider_impl).__module__
        is_native = any(provider_module.startswith(m) for m in self._NATIVE_INTERACTIONS_MODULES)

        if is_native and hasattr(provider_impl, "get_base_url"):
            base_url = str(provider_impl.get_base_url()).rstrip("/")
            # The Gemini provider returns a URL like
            # https://generativelanguage.googleapis.com/v1beta/openai
            # Strip the /openai suffix — Interactions sits alongside it at /v1beta/interactions
            if base_url.endswith("/openai"):
                base_url = base_url[: -len("/openai")]

            auth_headers: dict[str, str] = {}
            if hasattr(provider_impl, "get_passthrough_auth_headers"):
                auth_headers = provider_impl.get_passthrough_auth_headers()
            elif hasattr(provider_impl, "_get_api_key_from_config_or_provider_data"):
                api_key = provider_impl._get_api_key_from_config_or_provider_data()
                if api_key:
                    auth_headers = {"x-goog-api-key": api_key}

            if not auth_headers:
                logger.debug("No credentials for passthrough, falling back to translation", model=model)
                return None
            if provider_resource_id is None:
                logger.debug("No provider resource id for passthrough, falling back to translation", model=model)
                return None

            provider_config = getattr(provider_impl, "config", None)
            network_config = getattr(provider_config, "network", None)
            logger.info("Using native /interactions passthrough", model=model, base_url=base_url)
            return {
                "base_url": base_url,
                "auth_headers": auth_headers,
                "provider_resource_id": provider_resource_id,
                "network_config": network_config,
            }

        return None

    def _build_passthrough_client_kwargs(self, passthrough: _PassthroughInfo) -> dict[str, Any]:
        client_kwargs = build_network_client_kwargs(passthrough["network_config"])
        headers = dict(client_kwargs.get("headers", {}))
        headers["content-type"] = "application/json"
        headers.update(passthrough["auth_headers"])
        client_kwargs["headers"] = headers
        client_kwargs.setdefault("timeout", httpx.Timeout(300.0))
        return client_kwargs

    async def _passthrough_request(
        self,
        passthrough: _PassthroughInfo,
        request: GoogleCreateInteractionRequest,
    ) -> GoogleInteractionResponse | _RawSSEStream:
        """Forward the request directly to the provider's /interactions endpoint."""
        base_url = passthrough["base_url"]
        provider_model = passthrough["provider_resource_id"]

        url = f"{base_url}/interactions"
        body = request.model_dump(exclude_none=True)
        body["model"] = provider_model

        client_kwargs = self._build_passthrough_client_kwargs(passthrough)

        if request.stream:
            return self._passthrough_stream(url, body, client_kwargs)

        async with httpx.AsyncClient(**client_kwargs) as client:
            resp = await client.post(url, json=body)
            resp.raise_for_status()
            return GoogleInteractionResponse(**resp.json())

    def _passthrough_stream(
        self,
        url: str,
        body: dict[str, Any],
        client_kwargs: dict[str, Any],
    ) -> _RawSSEStream:
        """Stream raw SSE lines directly from the provider.

        Returns raw SSE-formatted strings instead of parsed event objects,
        preserving all event types (including thought events from thinking models).
        The returned iterator is marked with ``_raw_sse = True`` so the route
        layer can forward it without re-serialisation.
        """
        return _RawSSEStream(url, body, client_kwargs)

    def _parse_sse_event(self, event_type: str, data: dict[str, Any]) -> GoogleStreamEvent | None:
        """Parse a Google Interactions SSE event from its type and data."""
        if event_type == "interaction.start":
            return InteractionStartEvent(interaction=_InteractionRef(**data.get("interaction", data)))
        if event_type == "content.start":
            return ContentStartEvent(index=data.get("index", 0), content=_ContentRef())
        if event_type == "content.delta":
            delta_data = data.get("delta", {})
            return ContentDeltaEvent(index=data.get("index", 0), delta=_TextDelta(text=delta_data.get("text", "")))
        if event_type == "content.stop":
            return ContentStopEvent(index=data.get("index", 0))
        if event_type == "interaction.complete":
            return InteractionCompleteEvent(
                interaction=_InteractionCompleteRef(**data.get("interaction", data)),
            )
        return None

    # -- Request translation --

    def _google_to_openai(self, request: GoogleCreateInteractionRequest) -> OpenAIChatCompletionRequestWithExtraBody:
        messages = self._convert_input_to_openai(request.system_instruction, request.input)
        gen_config = request.generation_config or GoogleGenerationConfig()

        extra_body: dict[str, Any] = {}
        if gen_config.top_k is not None:
            extra_body["top_k"] = gen_config.top_k

        params = OpenAIChatCompletionRequestWithExtraBody(
            model=request.model,
            messages=messages,  # type: ignore[arg-type]
            max_tokens=gen_config.max_output_tokens,
            temperature=gen_config.temperature,
            top_p=gen_config.top_p,
            stream=request.stream or False,
            **(extra_body or {}),
        )
        return params

    def _convert_input_to_openai(
        self,
        system_instruction: str | None,
        input_data: str | list[GoogleInputTurn],
    ) -> list[dict[str, Any]]:
        openai_messages: list[dict[str, Any]] = []

        if system_instruction is not None:
            openai_messages.append({"role": "system", "content": system_instruction})

        if isinstance(input_data, str):
            openai_messages.append({"role": "user", "content": input_data})
        else:
            for turn in input_data:
                role = "assistant" if turn.role == "model" else turn.role
                # Extract text from content items
                text = "\n".join(item.text for item in turn.content)
                openai_messages.append({"role": role, "content": text})

        return openai_messages

    # -- Response translation --

    def _openai_to_google(self, response: OpenAIChatCompletion, request_model: str) -> GoogleInteractionResponse:
        outputs: list[GoogleTextOutput | GoogleThoughtOutput | GoogleOutput] = []

        if response.choices:
            choice = response.choices[0]
            message = choice.message

            if message and message.content:
                outputs.append(GoogleTextOutput(text=message.content))

        usage = GoogleUsage()
        if response.usage:
            input_tokens = response.usage.prompt_tokens or 0
            output_tokens = response.usage.completion_tokens or 0
            usage = GoogleUsage(
                total_input_tokens=input_tokens,
                total_output_tokens=output_tokens,
                total_tokens=input_tokens + output_tokens,
            )

        now = _now_iso()
        return GoogleInteractionResponse(
            id=f"interaction-{uuid.uuid4().hex[:24]}",
            created=now,
            updated=now,
            model=request_model,
            outputs=outputs,
            usage=usage,
        )

    # -- Streaming translation --

    async def _stream_openai_to_google(
        self,
        openai_stream: AsyncIterator[OpenAIChatCompletionChunk],
        request_model: str,
    ) -> AsyncIterator[GoogleStreamEvent]:
        """Translate OpenAI streaming chunks to Google streaming events."""

        interaction_id = f"interaction-{uuid.uuid4().hex[:24]}"

        # Emit interaction.start
        yield InteractionStartEvent(
            interaction=_InteractionRef(
                id=interaction_id,
                model=request_model,
            ),
        )

        content_started = False
        output_tokens = 0
        input_tokens = 0

        async for chunk in openai_stream:
            if not chunk.choices:
                # Usage-only chunk
                if chunk.usage:
                    input_tokens = chunk.usage.prompt_tokens or 0
                    output_tokens = chunk.usage.completion_tokens or 0
                continue

            choice = chunk.choices[0]
            delta = choice.delta

            if delta and delta.content:
                if not content_started:
                    yield ContentStartEvent(index=0, content=_ContentRef())
                    content_started = True

                yield ContentDeltaEvent(
                    index=0,
                    delta=_TextDelta(text=delta.content),
                )

            if chunk.usage:
                input_tokens = chunk.usage.prompt_tokens or 0
                output_tokens = chunk.usage.completion_tokens or 0

        # Close content block if opened
        if content_started:
            yield ContentStopEvent(index=0)

        # Final event
        now = _now_iso()
        yield InteractionCompleteEvent(
            interaction=_InteractionCompleteRef(
                id=interaction_id,
                created=now,
                updated=now,
                model=request_model,
                usage=GoogleUsage(
                    total_input_tokens=input_tokens,
                    total_output_tokens=output_tokens,
                    total_tokens=input_tokens + output_tokens,
                ),
            ),
        )
