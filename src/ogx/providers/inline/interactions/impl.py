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

import json
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, TypedDict

import httpx
from fastapi.responses import JSONResponse

from ogx.core.access_control.access_control import is_action_allowed
from ogx.core.access_control.conditions import User as AccessControlUser
from ogx.core.access_control.datatypes import Action
from ogx.core.datatypes import AccessRule
from ogx.core.request_headers import get_authenticated_user
from ogx.log import get_logger
from ogx.providers.utils.inference.http_client import build_network_client_kwargs
from ogx.providers.utils.inference.model_registry import NetworkConfig
from ogx.providers.utils.interactions.interactions_store import InteractionsStore
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
    GoogleContentItem,
    GoogleCreateInteractionRequest,
    GoogleFunctionCallContent,
    GoogleFunctionCallOutput,
    GoogleFunctionResponseContent,
    GoogleGenerationConfig,
    GoogleInputTurn,
    GoogleInteractionResponse,
    GoogleOutputItem,
    GoogleStreamEvent,
    GoogleTextContent,
    GoogleTextOutput,
    GoogleTool,
    GoogleUsage,
    InteractionCompleteEvent,
    InteractionStartEvent,
    _ContentRef,
    _FunctionCallContentRef,
    _FunctionCallDelta,
    _InteractionCompleteRef,
    _InteractionRef,
    _TextDelta,
)

from .config import InteractionsConfig

logger = get_logger(name=__name__, category="interactions")


def _now_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S+00:00")


def _now_epoch() -> int:
    return int(datetime.now(UTC).timestamp())


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

    def __init__(self, config: InteractionsConfig, inference_api: Inference, policy: list[AccessRule]):
        self.config = config
        self.inference_api = inference_api
        self.policy = policy

    async def initialize(self) -> None:
        self.store = InteractionsStore(self.config.store, self.policy)
        await self.store.initialize()

    async def shutdown(self) -> None:
        await self.store.shutdown()

    async def create_interaction(
        self,
        request: GoogleCreateInteractionRequest,
    ) -> GoogleInteractionResponse | AsyncIterator[GoogleStreamEvent] | AsyncIterator[str] | JSONResponse:
        passthrough = await self._get_passthrough_info(request.model)
        if passthrough:
            return await self._passthrough_request(passthrough, request)

        messages = await self._build_messages(request)
        openai_params = self._google_to_openai(request, messages)

        result = await self.inference_api.openai_chat_completion(openai_params)

        if isinstance(result, AsyncIterator):
            return self._stream_openai_to_google(result, request.model, messages)

        return await self._openai_to_google(result, request.model, messages)

    async def _build_messages(self, request: GoogleCreateInteractionRequest) -> list[dict[str, Any]]:
        """Build the full message list, incorporating previous interaction context if chaining."""
        if request.previous_interaction_id:
            stored = await self.store.get_interaction(request.previous_interaction_id)
            if not stored:
                raise ValueError(
                    f"Interaction '{request.previous_interaction_id}' not found. "
                    "Cannot chain from a non-existent interaction."
                )
            messages: list[dict[str, Any]] = list(stored["messages"])
            messages.append({"role": "assistant", "content": stored["output_text"]})
            messages.extend(self._convert_input_to_openai(None, request.input))
            return messages

        return self._convert_input_to_openai(request.system_instruction, request.input)

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
    ) -> GoogleInteractionResponse | _RawSSEStream | JSONResponse:
        """Forward the request directly to the provider's /interactions endpoint."""
        base_url = passthrough["base_url"]
        provider_model = passthrough["provider_resource_id"]

        url = f"{base_url}/interactions"
        body = request.model_dump(exclude_none=True, by_alias=True)
        body["model"] = provider_model

        # Transform tools from function_declarations format to Interactions API format.
        # generateContent uses: [{"function_declarations": [{name, ...}]}]
        # Interactions API uses: [{"type": "function", "function": {name, ...}}]
        if "tools" in body and body["tools"]:
            body["tools"] = self._transform_tools_for_interactions_api(body["tools"])

        client_kwargs = self._build_passthrough_client_kwargs(passthrough)

        if request.stream:
            return self._passthrough_stream(url, body, client_kwargs)

        async with httpx.AsyncClient(**client_kwargs) as client:
            resp = await client.post(url, json=body)
            if resp.status_code >= 400:
                logger.error(
                    "Passthrough request failed",
                    status_code=resp.status_code,
                    response_body=resp.text[:500],
                    url=url,
                )
                return JSONResponse(content=resp.json(), status_code=resp.status_code)
            # Forward the raw JSON to preserve all provider-specific fields
            return JSONResponse(content=resp.json(), status_code=resp.status_code)

    @staticmethod
    def _transform_tools_for_interactions_api(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Transform tools from function_declarations format to Interactions API format.

        The generateContent API uses: [{"function_declarations": [{name, ...}]}]
        The Interactions API uses: [{"type": "function", "name": ..., "description": ..., ...}]
        Each function_declaration becomes a separate flat tool entry.
        """
        result: list[dict[str, Any]] = []
        for tool in tools:
            declarations = tool.get("function_declarations")
            if declarations:
                for decl in declarations:
                    entry = {"type": "function"}
                    entry.update(decl)
                    result.append(entry)
            else:
                # Already in Interactions API format or unknown — pass through
                result.append(tool)
        return result

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

    def _google_to_openai(
        self,
        request: GoogleCreateInteractionRequest,
        messages: list[dict[str, Any]],
    ) -> OpenAIChatCompletionRequestWithExtraBody:
        gen_config = request.generation_config or GoogleGenerationConfig()

        extra_body: dict[str, Any] = {}
        if gen_config.top_k is not None:
            extra_body["top_k"] = gen_config.top_k

        tools = self._convert_tools_to_openai(request.tools) if request.tools else None

        params = OpenAIChatCompletionRequestWithExtraBody(
            model=request.model,
            messages=messages,  # type: ignore[arg-type]
            max_tokens=gen_config.max_output_tokens,
            temperature=gen_config.temperature,
            top_p=gen_config.top_p,
            stream=request.stream or False,
            tools=tools,  # type: ignore[arg-type]
            **(extra_body or {}),
        )
        return params

    def _convert_tools_to_openai(self, tools: list[GoogleTool]) -> list[dict[str, Any]]:
        openai_tools: list[dict[str, Any]] = []
        for tool in tools:
            for decl in tool.function_declarations:
                openai_tool: dict[str, Any] = {
                    "type": "function",
                    "function": {
                        "name": decl.name,
                        "description": decl.description or "",
                    },
                }
                if decl.parameters:
                    openai_tool["function"]["parameters"] = decl.parameters
                openai_tools.append(openai_tool)
        return openai_tools

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
                openai_messages.extend(self._convert_turn_to_openai(turn))

        return openai_messages

    def _convert_turn_to_openai(self, turn: GoogleInputTurn) -> list[dict[str, Any]]:
        role = "assistant" if turn.role == "model" else turn.role

        # Handle string content (SDK may send content as a plain string)
        if isinstance(turn.content, str):
            return [{"role": role, "content": turn.content}]

        # Check for function_call or function_response content
        has_function_calls = any(isinstance(item, GoogleFunctionCallContent) for item in turn.content)
        has_function_responses = any(isinstance(item, GoogleFunctionResponseContent) for item in turn.content)

        if has_function_calls and role == "assistant":
            return [self._convert_model_turn_with_function_calls(turn.content)]

        if has_function_responses:
            return self._convert_turn_with_function_responses(turn.content)

        # Text-only turn
        text_parts = [item.text for item in turn.content if isinstance(item, GoogleTextContent)]
        text = "\n".join(text_parts)
        return [{"role": role, "content": text}]

    def _convert_model_turn_with_function_calls(
        self,
        content: list[GoogleContentItem],
    ) -> dict[str, Any]:
        msg: dict[str, Any] = {"role": "assistant"}
        text_parts: list[str] = []
        tool_calls: list[dict[str, Any]] = []

        for item in content:
            if isinstance(item, GoogleTextContent):
                text_parts.append(item.text)
            elif isinstance(item, GoogleFunctionCallContent):
                tool_calls.append(
                    {
                        "id": item.id or f"call_{uuid.uuid4().hex[:24]}",
                        "type": "function",
                        "function": {
                            "name": item.name,
                            "arguments": json.dumps(item.args),
                        },
                    }
                )

        if text_parts:
            msg["content"] = "\n".join(text_parts)
        if tool_calls:
            msg["tool_calls"] = tool_calls

        return msg

    def _convert_turn_with_function_responses(
        self,
        content: list[GoogleContentItem],
    ) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        text_parts: list[str] = []

        for item in content:
            if isinstance(item, GoogleTextContent):
                text_parts.append(item.text)
            elif isinstance(item, GoogleFunctionResponseContent):
                # Flush any accumulated text as a user message
                if text_parts:
                    messages.append({"role": "user", "content": "\n".join(text_parts)})
                    text_parts = []
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": item.call_id or item.id or "",
                        "content": json.dumps(item.response),
                    }
                )

        if text_parts:
            messages.append({"role": "user", "content": "\n".join(text_parts)})

        return messages

    # -- Response translation --

    async def _openai_to_google(
        self,
        response: OpenAIChatCompletion,
        request_model: str,
        messages: list[dict[str, Any]],
    ) -> GoogleInteractionResponse:
        outputs: list[GoogleOutputItem] = []

        if response.choices:
            choice = response.choices[0]
            message = choice.message

            if message and message.content:
                outputs.append(GoogleTextOutput(text=message.content))

            if message and message.tool_calls:
                for tc in message.tool_calls:
                    if not hasattr(tc, "function") or tc.function is None:
                        continue
                    try:
                        args = json.loads(tc.function.arguments) if tc.function.arguments else {}
                    except json.JSONDecodeError:
                        args = {}
                    outputs.append(
                        GoogleFunctionCallOutput(
                            id=tc.id or f"call_{uuid.uuid4().hex[:24]}",
                            name=tc.function.name or "",
                            args=args,
                        )
                    )

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
        interaction_id = f"interaction-{uuid.uuid4().hex[:24]}"
        output_text = ""
        for o in outputs:
            if isinstance(o, GoogleTextOutput):
                output_text = o.text
                break

        await self.store.store_interaction(
            interaction_id=interaction_id,
            created_at=_now_epoch(),
            model=request_model,
            messages=messages,
            output_text=output_text,
        )

        return GoogleInteractionResponse(
            id=interaction_id,
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
        messages: list[dict[str, Any]],
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

        content_block_index = 0
        text_block_started = False
        collected_text: list[str] = []
        tool_call_blocks: dict[int, bool] = {}  # tc_index -> started
        tool_call_index_to_block_index: dict[int, int] = {}
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
                if not text_block_started:
                    yield ContentStartEvent(index=content_block_index, content=_ContentRef())
                    text_block_started = True

                collected_text.append(delta.content)
                yield ContentDeltaEvent(
                    index=content_block_index,
                    delta=_TextDelta(text=delta.content),
                )

            if delta and delta.tool_calls:
                for tc_delta in delta.tool_calls:
                    tc_idx = tc_delta.index if tc_delta.index is not None else 0

                    if tc_idx not in tool_call_blocks:
                        # Close text block if open
                        if text_block_started:
                            yield ContentStopEvent(index=content_block_index)
                            content_block_index += 1
                            text_block_started = False

                        # Start new function_call block
                        tool_call_blocks[tc_idx] = True
                        tool_call_index_to_block_index[tc_idx] = content_block_index

                        yield ContentStartEvent(
                            index=content_block_index,
                            content=_FunctionCallContentRef(
                                id=tc_delta.id or f"call_{uuid.uuid4().hex[:24]}",
                                name=tc_delta.function.name if tc_delta.function and tc_delta.function.name else None,
                            ),
                        )
                        content_block_index += 1

                    if tc_delta.function and tc_delta.function.arguments:
                        block_idx = tool_call_index_to_block_index[tc_idx]
                        yield ContentDeltaEvent(
                            index=block_idx,
                            delta=_FunctionCallDelta(args=tc_delta.function.arguments),
                        )

            if chunk.usage:
                input_tokens = chunk.usage.prompt_tokens or 0
                output_tokens = chunk.usage.completion_tokens or 0

        # Close any open blocks
        if text_block_started:
            yield ContentStopEvent(index=content_block_index)

        for _tc_idx, block_idx in tool_call_index_to_block_index.items():
            yield ContentStopEvent(index=block_idx)

        # Store the interaction for conversation chaining
        await self.store.store_interaction(
            interaction_id=interaction_id,
            created_at=_now_epoch(),
            model=request_model,
            messages=messages,
            output_text="".join(collected_text),
        )

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
