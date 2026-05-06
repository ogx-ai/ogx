# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Built-in Anthropic Messages API implementation.

Translates Anthropic Messages format to/from OpenAI Chat Completions format,
delegating to the inference API for actual model calls. When the underlying
inference provider natively supports the Anthropic Messages API (e.g. Ollama),
requests are forwarded directly without translation.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator
from typing import Any

import httpx

from ogx.log import get_logger
from ogx_api import (
    Inference,
    OpenAIChatCompletion,
    OpenAIChatCompletionChunk,
    OpenAIChatCompletionRequestWithExtraBody,
)
from ogx_api.messages import (
    Messages,
)
from ogx_api.messages.models import (
    ANTHROPIC_VERSION,
    AnthropicContentBlock,
    AnthropicCountTokensRequest,
    AnthropicCountTokensResponse,
    AnthropicCreateMessageRequest,
    AnthropicImageBlock,
    AnthropicMessage,
    AnthropicMessageResponse,
    AnthropicStreamEvent,
    AnthropicTextBlock,
    AnthropicThinkingBlock,
    AnthropicToolDef,
    AnthropicToolResultBlock,
    AnthropicToolUseBlock,
    AnthropicUsage,
    ContentBlockDeltaEvent,
    ContentBlockStartEvent,
    ContentBlockStopEvent,
    MessageDeltaEvent,
    MessageStartEvent,
    MessageStopEvent,
    _InputJsonDelta,
    _MessageDelta,
    _TextDelta,
    _ThinkingDelta,
)

from .config import MessagesConfig

logger = get_logger(name=__name__, category="messages")

# Maps Anthropic stop_reason -> OpenAI finish_reason
_STOP_REASON_TO_FINISH = {
    "end_turn": "stop",
    "stop_sequence": "stop",
    "tool_use": "tool_calls",
    "max_tokens": "length",
}

# Maps OpenAI finish_reason -> Anthropic stop_reason
_FINISH_TO_STOP_REASON = {
    "stop": "end_turn",
    "tool_calls": "tool_use",
    "length": "max_tokens",
    "content_filter": "end_turn",
}


class BuiltinMessagesImpl(Messages):
    """Anthropic Messages API adapter that translates to the inference API."""

    def __init__(self, config: MessagesConfig, inference_api: Inference):
        self.config = config
        self.inference_api = inference_api

    async def initialize(self) -> None:
        self._client = httpx.AsyncClient()

    async def shutdown(self) -> None:
        await self._client.aclose()

    async def create_message(
        self,
        request: AnthropicCreateMessageRequest,
    ) -> AnthropicMessageResponse | AsyncIterator[AnthropicStreamEvent]:
        # Try native passthrough for providers that support /v1/messages directly
        passthrough_url = await self._get_passthrough_url(request.model)
        if passthrough_url:
            return await self._passthrough_request(passthrough_url, request)

        # Translation mode: convert Anthropic format to OpenAI format
        openai_params = self._anthropic_to_openai(request)
        openai_result = await self.inference_api.openai_chat_completion(openai_params)

        if isinstance(openai_result, AsyncIterator):
            return self._stream_openai_to_anthropic(openai_result, request.model)

        return self._openai_to_anthropic(openai_result, request.model)

    async def count_message_tokens(
        self,
        request: AnthropicCountTokensRequest,
    ) -> AnthropicCountTokensResponse:
        passthrough_url = await self._get_passthrough_url(request.model)
        if passthrough_url:
            return await self._passthrough_count_tokens(passthrough_url, request)

        # Translation mode: use Inference API's count_tokens if available
        raise NotImplementedError("Token counting via translation mode is not yet implemented")

    # -- Native passthrough for providers with /v1/messages support --

    # Module paths of provider impls known to support /v1/messages natively
    _NATIVE_MESSAGES_MODULES = {
        "ogx.providers.remote.inference.ollama",
        "ogx.providers.remote.inference.vllm",
    }

    async def _get_passthrough_url(self, model: str) -> str | None:
        """Check if the model's provider supports /v1/messages natively.

        Returns the base URL for passthrough, or None to use translation.
        """
        router = self.inference_api
        if not hasattr(router, "routing_table"):
            return None

        try:
            obj = await router.routing_table.get_object_by_identifier("model", model)
            if not obj:
                return None

            provider_impl = await router.routing_table.get_provider_impl(obj.identifier)
            provider_module = type(provider_impl).__module__
            is_native = any(provider_module.startswith(m) for m in self._NATIVE_MESSAGES_MODULES)

            if is_native and hasattr(provider_impl, "get_base_url"):
                base_url = str(provider_impl.get_base_url()).rstrip("/")
                # Ollama's /v1/messages sits at the root, not under /v1
                if base_url.endswith("/v1"):
                    base_url = base_url[:-3]
                logger.info("Using native /v1/messages passthrough", model=model, base_url=base_url)
                return base_url
        except Exception:
            logger.debug("Failed to resolve passthrough, falling back to translation", model=model)

        return None

    async def _passthrough_request(
        self,
        base_url: str,
        request: AnthropicCreateMessageRequest,
    ) -> AnthropicMessageResponse | AsyncIterator[AnthropicStreamEvent]:
        """Forward the request directly to the provider's /v1/messages endpoint."""
        url = f"{base_url}/v1/messages"
        # Use the provider_resource_id (model name without provider prefix)
        provider_model = request.model
        router = self.inference_api
        if hasattr(router, "routing_table"):
            try:
                obj = await router.routing_table.get_object_by_identifier("model", request.model)
                if obj:
                    provider_model = obj.provider_resource_id
            except Exception:
                pass

        body = request.model_dump(exclude_none=True)
        body["model"] = provider_model
        headers = {
            "content-type": "application/json",
            "anthropic-version": ANTHROPIC_VERSION,
            "x-api-key": "no-key-required",
        }

        if request.stream:
            return self._passthrough_stream(url, headers, body)

        resp = await self._client.post(url, json=body, headers=headers, timeout=300)
        resp.raise_for_status()
        return AnthropicMessageResponse(**resp.json())

    async def _passthrough_stream(
        self,
        url: str,
        headers: dict[str, str],
        body: dict[str, Any],
    ) -> AsyncIterator[AnthropicStreamEvent]:
        """Stream SSE events directly from the provider."""
        async with self._client.stream("POST", url, json=body, headers=headers, timeout=300) as resp:
            resp.raise_for_status()
            event_type = None
            async for line in resp.aiter_lines():
                line = line.strip()
                if line.startswith("event: "):
                    event_type = line[7:]
                elif line.startswith("data: ") and event_type:
                    data = json.loads(line[6:])
                    event = self._parse_sse_event(event_type, data)
                    if event:
                        yield event
                    event_type = None

    def _parse_sse_event(self, event_type: str, data: dict[str, Any]) -> AnthropicStreamEvent | None:
        """Parse an Anthropic SSE event from its type and data."""
        if event_type == "message_start":
            return MessageStartEvent(message=AnthropicMessageResponse(**data["message"]))
        if event_type == "content_block_start":
            block_data = data["content_block"]
            content_block: AnthropicTextBlock | AnthropicToolUseBlock | AnthropicThinkingBlock
            block_type = block_data.get("type")
            if block_type == "tool_use":
                content_block = AnthropicToolUseBlock(**block_data)
            elif block_type == "thinking":
                content_block = AnthropicThinkingBlock(**block_data)
            else:
                content_block = AnthropicTextBlock(**block_data)
            return ContentBlockStartEvent(index=data["index"], content_block=content_block)
        if event_type == "content_block_delta":
            delta_data = data["delta"]
            delta_type = delta_data.get("type")
            delta: _TextDelta | _InputJsonDelta | _ThinkingDelta
            if delta_type == "text_delta":
                delta = _TextDelta(text=delta_data["text"])
            elif delta_type == "input_json_delta":
                delta = _InputJsonDelta(partial_json=delta_data["partial_json"])
            elif delta_type == "thinking_delta":
                delta = _ThinkingDelta(thinking=delta_data["thinking"])
            else:
                return None
            return ContentBlockDeltaEvent(index=data["index"], delta=delta)
        if event_type == "content_block_stop":
            return ContentBlockStopEvent(index=data["index"])
        if event_type == "message_delta":
            return MessageDeltaEvent(
                delta=_MessageDelta(stop_reason=data["delta"].get("stop_reason")),
                usage=AnthropicUsage(**data.get("usage", {})),
            )
        if event_type == "message_stop":
            return MessageStopEvent()
        return None

    async def _passthrough_count_tokens(
        self,
        base_url: str,
        request: AnthropicCountTokensRequest,
    ) -> AnthropicCountTokensResponse:
        """Forward the count_tokens request to the provider's /v1/messages/count_tokens endpoint."""
        url = f"{base_url}/v1/messages/count_tokens"
        # Use the provider_resource_id (model name without provider prefix)
        provider_model = request.model
        router = self.inference_api
        if hasattr(router, "routing_table"):
            try:
                obj = await router.routing_table.get_object_by_identifier("model", request.model)
                if obj:
                    provider_model = obj.provider_resource_id
            except Exception:
                pass

        body = request.model_dump(exclude_none=True)
        body["model"] = provider_model
        headers = {
            "content-type": "application/json",
            "anthropic-version": ANTHROPIC_VERSION,
            "x-api-key": "no-key-required",
        }

        resp = await self._client.post(url, json=body, headers=headers, timeout=30)
        resp.raise_for_status()
        return AnthropicCountTokensResponse(**resp.json())

    # -- Request translation --

    def _anthropic_to_openai(self, request: AnthropicCreateMessageRequest) -> OpenAIChatCompletionRequestWithExtraBody:
        messages = self._convert_messages_to_openai(request.system, request.messages)
        tools = self._convert_tools_to_openai(request.tools) if request.tools else None
        tool_choice = self._convert_tool_choice_to_openai(request.tool_choice) if request.tool_choice else None

        extra_body: dict[str, Any] = {}
        if request.top_k is not None:
            extra_body["top_k"] = request.top_k
        # Note: Anthropic's "thinking" parameter has no equivalent in the OpenAI
        # chat completions API and is intentionally not forwarded.

        params = OpenAIChatCompletionRequestWithExtraBody(
            model=request.model,
            messages=messages,  # type: ignore[arg-type]
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            stop=request.stop_sequences,
            tools=tools,
            tool_choice=tool_choice,
            stream=request.stream or False,
            service_tier=request.service_tier,  # type: ignore[arg-type]
            **(extra_body or {}),
        )
        return params

    def _convert_messages_to_openai(
        self,
        system: str | list[AnthropicTextBlock] | None,
        messages: list[AnthropicMessage],
    ) -> list[dict[str, Any]]:
        openai_messages: list[dict[str, Any]] = []

        if system is not None:
            if isinstance(system, str):
                system_text = system
            else:
                system_text = "\n".join(block.text for block in system)
            openai_messages.append({"role": "system", "content": system_text})

        for msg in messages:
            openai_messages.extend(self._convert_single_message(msg))

        return openai_messages

    def _convert_single_message(self, msg: AnthropicMessage) -> list[dict[str, Any]]:
        """Convert a single Anthropic message to one or more OpenAI messages.

        A single Anthropic user message with tool_result blocks may need to be
        split into multiple OpenAI messages (tool messages).
        """
        if isinstance(msg.content, str):
            return [{"role": msg.role, "content": msg.content}]

        if msg.role == "assistant":
            return [self._convert_assistant_message(msg.content)]

        # User message: may contain text and/or tool_result blocks
        result: list[dict[str, Any]] = []
        text_parts: list[dict[str, Any]] = []

        for block in msg.content:
            if isinstance(block, AnthropicToolResultBlock):
                # Flush accumulated text first
                if text_parts:
                    if len(text_parts) == 1 and text_parts[0].get("type") == "text":
                        flush_content: str | list[dict[str, Any]] = text_parts[0]["text"]
                    else:
                        flush_content = text_parts
                    result.append({"role": "user", "content": flush_content})
                    text_parts = []
                # Tool results become separate tool messages
                tool_content = block.content
                if isinstance(tool_content, list):
                    tool_content = "\n".join(b.text for b in tool_content if isinstance(b, AnthropicTextBlock))
                result.append(
                    {
                        "role": "tool",
                        "tool_call_id": block.tool_use_id,
                        "content": tool_content,
                    }
                )
            elif isinstance(block, AnthropicTextBlock):
                text_parts.append({"type": "text", "text": block.text})
            elif isinstance(block, AnthropicImageBlock):
                text_parts.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{block.source.media_type};base64,{block.source.data}",
                        },
                    }
                )

        if text_parts:
            # OpenAI content must be a string or a list, never a single dict
            if len(text_parts) == 1 and text_parts[0].get("type") == "text":
                user_content: str | list[dict[str, Any]] = text_parts[0]["text"]
            else:
                user_content = text_parts
            result.append({"role": "user", "content": user_content})

        return result if result else [{"role": "user", "content": ""}]

    def _convert_assistant_message(self, content: list[AnthropicContentBlock]) -> dict[str, Any]:
        """Convert an assistant message with content blocks to OpenAI format."""
        text_parts: list[str] = []
        tool_calls: list[dict[str, Any]] = []

        for block in content:
            if isinstance(block, AnthropicTextBlock):
                text_parts.append(block.text)
            elif isinstance(block, AnthropicToolUseBlock):
                tool_calls.append(
                    {
                        "id": block.id,
                        "type": "function",
                        "function": {
                            "name": block.name,
                            "arguments": json.dumps(block.input),
                        },
                    }
                )

        msg: dict[str, Any] = {"role": "assistant"}
        if text_parts:
            msg["content"] = "\n".join(text_parts)
        if tool_calls:
            msg["tool_calls"] = tool_calls

        return msg

    def _convert_tools_to_openai(self, tools: list[AnthropicToolDef]) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description or "",
                    "parameters": tool.input_schema,
                },
            }
            for tool in tools
        ]

    def _convert_tool_choice_to_openai(self, tool_choice: Any) -> Any:
        if isinstance(tool_choice, str):
            if tool_choice == "any":
                return "required"
            if tool_choice == "none":
                return "none"
            return "auto"

        if isinstance(tool_choice, dict):
            tc_type = tool_choice.get("type")
            if tc_type == "tool":
                return {"type": "function", "function": {"name": tool_choice["name"]}}
            if tc_type == "any":
                return "required"
            if tc_type == "none":
                return "none"
            return "auto"

        return "auto"

    # -- Response translation --

    def _openai_to_anthropic(self, response: OpenAIChatCompletion, request_model: str) -> AnthropicMessageResponse:
        content: list[AnthropicContentBlock] = []

        if response.choices:
            choice = response.choices[0]
            message = choice.message

            if message and message.content:
                content.append(AnthropicTextBlock(text=message.content))

            if message and message.tool_calls:
                for tc in message.tool_calls:
                    if not hasattr(tc, "function") or tc.function is None:
                        continue
                    try:
                        tool_input = json.loads(tc.function.arguments) if tc.function.arguments else {}
                    except json.JSONDecodeError:
                        tool_input = {}

                    content.append(
                        AnthropicToolUseBlock(
                            id=tc.id or f"toolu_{uuid.uuid4().hex[:24]}",
                            name=tc.function.name or "",
                            input=tool_input,
                        )
                    )

            finish_reason = choice.finish_reason or "stop"
            stop_reason = _FINISH_TO_STOP_REASON.get(finish_reason, "end_turn")
        else:
            stop_reason = "end_turn"

        usage = AnthropicUsage()
        if response.usage:
            cache_read = None
            if response.usage.prompt_tokens_details and hasattr(response.usage.prompt_tokens_details, "cached_tokens"):
                cache_read = response.usage.prompt_tokens_details.cached_tokens

            usage = AnthropicUsage(
                input_tokens=response.usage.prompt_tokens or 0,
                output_tokens=response.usage.completion_tokens or 0,
                cache_read_input_tokens=cache_read,
            )

        return AnthropicMessageResponse(
            id=f"msg_{uuid.uuid4().hex[:24]}",
            content=content,
            model=request_model,
            stop_reason=stop_reason,
            usage=usage,
        )

    # -- Streaming translation --

    async def _stream_openai_to_anthropic(
        self,
        openai_stream: AsyncIterator[OpenAIChatCompletionChunk],
        request_model: str,
    ) -> AsyncIterator[AnthropicStreamEvent]:
        """Translate OpenAI streaming chunks to Anthropic streaming events."""

        # Emit message_start
        yield MessageStartEvent(
            message=AnthropicMessageResponse(
                id=f"msg_{uuid.uuid4().hex[:24]}",
                content=[],
                model=request_model,
                stop_reason=None,
                usage=AnthropicUsage(input_tokens=0, output_tokens=0),
            ),
        )

        content_block_index = 0
        in_text_block = False
        in_tool_blocks: dict[int, bool] = {}  # tool_call_index -> started
        tool_call_index_to_block_index: dict[int, int] = {}
        output_tokens = 0
        input_tokens = 0
        cache_read_tokens: int | None = None
        stop_reason = "end_turn"

        async for chunk in openai_stream:
            if not chunk.choices:
                # Usage-only chunk
                if chunk.usage:
                    input_tokens = chunk.usage.prompt_tokens or 0
                    output_tokens = chunk.usage.completion_tokens or 0
                    if chunk.usage.prompt_tokens_details and hasattr(
                        chunk.usage.prompt_tokens_details, "cached_tokens"
                    ):
                        cache_read_tokens = chunk.usage.prompt_tokens_details.cached_tokens
                continue

            choice = chunk.choices[0]
            delta = choice.delta

            if delta and delta.content:
                if not in_text_block:
                    yield ContentBlockStartEvent(
                        index=content_block_index,
                        content_block=AnthropicTextBlock(text=""),
                    )
                    in_text_block = True

                yield ContentBlockDeltaEvent(
                    index=content_block_index,
                    delta=_TextDelta(text=delta.content),
                )

            if delta and delta.tool_calls:
                for tc_delta in delta.tool_calls:
                    tc_idx = tc_delta.index if tc_delta.index is not None else 0

                    if tc_idx not in in_tool_blocks:
                        # Close text block if open
                        if in_text_block:
                            yield ContentBlockStopEvent(index=content_block_index)
                            content_block_index += 1
                            in_text_block = False

                        # Start new tool_use block
                        in_tool_blocks[tc_idx] = True
                        tool_call_index_to_block_index[tc_idx] = content_block_index

                        yield ContentBlockStartEvent(
                            index=content_block_index,
                            content_block=AnthropicToolUseBlock(
                                id=tc_delta.id or f"toolu_{uuid.uuid4().hex[:24]}",
                                name=tc_delta.function.name if tc_delta.function and tc_delta.function.name else "",
                                input={},
                            ),
                        )
                        content_block_index += 1

                    if tc_delta.function and tc_delta.function.arguments:
                        block_idx = tool_call_index_to_block_index[tc_idx]
                        yield ContentBlockDeltaEvent(
                            index=block_idx,
                            delta=_InputJsonDelta(partial_json=tc_delta.function.arguments),
                        )

            if choice.finish_reason:
                stop_reason = _FINISH_TO_STOP_REASON.get(choice.finish_reason, "end_turn")

            if chunk.usage:
                input_tokens = chunk.usage.prompt_tokens or 0
                output_tokens = chunk.usage.completion_tokens or 0
                if chunk.usage.prompt_tokens_details and hasattr(chunk.usage.prompt_tokens_details, "cached_tokens"):
                    cache_read_tokens = chunk.usage.prompt_tokens_details.cached_tokens

        # Close any open blocks
        if in_text_block:
            yield ContentBlockStopEvent(index=content_block_index)

        for _tc_idx, block_idx in tool_call_index_to_block_index.items():
            yield ContentBlockStopEvent(index=block_idx)

        # Final events
        yield MessageDeltaEvent(
            delta=_MessageDelta(stop_reason=stop_reason),
            usage=AnthropicUsage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cache_read_input_tokens=cache_read_tokens,
            ),
        )
        yield MessageStopEvent()
