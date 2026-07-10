# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Anthropic Messages to/from OpenAI Chat Completions translation utilities.

Shared by:
- OpenAIMixin (default translation implementation)
- Native passthrough providers (Ollama, vLLM) for SSE event parsing
"""

from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator
from typing import Any

from ogx.log import get_logger
from ogx_api import (
    OpenAIChatCompletion,
    OpenAIChatCompletionChunk,
    OpenAIChatCompletionRequestWithExtraBody,
)
from ogx_api.messages.models import (
    AnthropicBase64ImageSource,
    AnthropicContentBlock,
    AnthropicCreateMessageRequest,
    AnthropicCustomToolDef,
    AnthropicImageBlock,
    AnthropicMessage,
    AnthropicMessageResponse,
    AnthropicRedactedThinkingBlock,
    AnthropicStreamEvent,
    AnthropicTextBlock,
    AnthropicThinkingBlock,
    AnthropicTool,
    AnthropicToolResultBlock,
    AnthropicToolUseBlock,
    AnthropicURLImageSource,
    AnthropicUsage,
    ContentBlockDeltaEvent,
    ContentBlockStartEvent,
    ContentBlockStopEvent,
    ErrorStreamEvent,
    MessageDeltaEvent,
    MessageStartEvent,
    MessageStopEvent,
    PingEvent,
    _AnthropicErrorDetail,
    _InputJsonDelta,
    _MessageDelta,
    _SignatureDelta,
    _TextDelta,
    _ThinkingDelta,
    _ToolChoiceTool,
)

logger = get_logger(name=__name__, category="providers::utils")

# Maps Anthropic stop_reason -> OpenAI finish_reason
_STOP_REASON_TO_FINISH = {
    "end_turn": "stop",
    "stop_sequence": "stop",
    "tool_use": "tool_calls",
    "max_tokens": "length",
    "pause_turn": "stop",
}

# Maps OpenAI finish_reason -> Anthropic stop_reason
_FINISH_TO_STOP_REASON = {
    "stop": "end_turn",
    "tool_calls": "tool_use",
    "length": "max_tokens",
    "content_filter": "end_turn",
}


def _image_source_to_url(source: AnthropicBase64ImageSource | AnthropicURLImageSource) -> str:
    if isinstance(source, AnthropicBase64ImageSource):
        return f"data:{source.media_type};base64,{source.data}"
    return source.url


def convert_messages_to_openai(
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
        openai_messages.extend(convert_single_message(msg))

    return openai_messages


def convert_single_message(msg: AnthropicMessage) -> list[dict[str, Any]]:
    """Convert a single Anthropic message to one or more OpenAI messages.

    A single Anthropic user message with tool_result blocks may need to be
    split into multiple OpenAI messages (tool messages).
    """
    if isinstance(msg.content, str):
        return [{"role": msg.role, "content": msg.content}]

    if msg.role == "assistant":
        return [convert_assistant_message(msg.content)]

    if msg.role == "system":
        system_text = "\n".join(block.text for block in msg.content if isinstance(block, AnthropicTextBlock))
        return [{"role": "system", "content": system_text}]

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
            # Tool results become separate tool messages.
            # OpenAI tool messages only support text content, so image blocks
            # from tool results are promoted to a follow-up user message.
            tool_content = block.content
            image_parts: list[dict[str, Any]] = []
            if isinstance(tool_content, list):
                text_pieces = []
                for b in tool_content:
                    if isinstance(b, AnthropicTextBlock):
                        text_pieces.append(b.text)
                    elif isinstance(b, AnthropicImageBlock):
                        image_parts.append({"type": "image_url", "image_url": {"url": _image_source_to_url(b.source)}})
                tool_content = "\n".join(text_pieces)
            result.append(
                {
                    "role": "tool",
                    "tool_call_id": block.tool_use_id,
                    "content": tool_content,
                }
            )
            if image_parts:
                result.append({"role": "user", "content": image_parts})
        elif isinstance(block, AnthropicTextBlock):
            text_parts.append({"type": "text", "text": block.text})
        elif isinstance(block, AnthropicImageBlock):
            text_parts.append({"type": "image_url", "image_url": {"url": _image_source_to_url(block.source)}})

    if text_parts:
        if len(text_parts) == 1 and text_parts[0].get("type") == "text":
            user_content: str | list[dict[str, Any]] = text_parts[0]["text"]
        else:
            user_content = text_parts
        result.append({"role": "user", "content": user_content})

    return result if result else [{"role": "user", "content": ""}]


def convert_assistant_message(content: list[AnthropicContentBlock]) -> dict[str, Any]:
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


def convert_tools_to_openai(tools: list[AnthropicTool]) -> list[dict[str, Any]] | None:
    result = []
    for tool in tools:
        if not isinstance(tool, AnthropicCustomToolDef):
            logger.debug("Dropping server-side tool in translation mode", tool_type=tool.type)
            continue
        result.append(
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description or "",
                    "parameters": tool.input_schema,
                },
            }
        )
    return result or None


def convert_tool_choice_to_openai(tool_choice: Any) -> Any:
    if isinstance(tool_choice, _ToolChoiceTool):
        return {"type": "function", "function": {"name": tool_choice.name}}

    tc_type = tool_choice.type if hasattr(tool_choice, "type") else str(tool_choice)
    if tc_type == "any":
        return "required"
    if tc_type == "none":
        return "none"
    return "auto"


def anthropic_request_to_openai(request: AnthropicCreateMessageRequest) -> OpenAIChatCompletionRequestWithExtraBody:
    """Convert an Anthropic CreateMessage request to OpenAI chat completion params."""
    if request.thinking and request.thinking.type == "enabled":
        raise ValueError(
            "Failed to process thinking request: extended thinking requires a native "
            "Anthropic-compatible provider; translation mode does not support it"
        )

    messages = convert_messages_to_openai(request.system, request.messages)
    tools = convert_tools_to_openai(request.tools) if request.tools else None
    tool_choice = convert_tool_choice_to_openai(request.tool_choice) if tools and request.tool_choice else None

    parallel_tool_calls: bool | None = None
    if request.tool_choice and getattr(request.tool_choice, "disable_parallel_tool_use", False):
        parallel_tool_calls = False

    extra_body: dict[str, Any] = {}
    if request.top_k is not None:
        extra_body["top_k"] = request.top_k

    return OpenAIChatCompletionRequestWithExtraBody(
        model=request.model,
        messages=messages,  # type: ignore[arg-type]
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        stop=request.stop_sequences,
        tools=tools,
        tool_choice=tool_choice,
        parallel_tool_calls=parallel_tool_calls,
        stream=request.stream or False,
        service_tier=request.service_tier,  # type: ignore[arg-type]
        **(extra_body or {}),
    )


def openai_response_to_anthropic(response: OpenAIChatCompletion, request_model: str) -> AnthropicMessageResponse:
    """Convert an OpenAI ChatCompletion response to Anthropic message format."""
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


async def openai_stream_to_anthropic(
    openai_stream: AsyncIterator[OpenAIChatCompletionChunk],
    request_model: str,
) -> AsyncIterator[AnthropicStreamEvent]:
    """Translate OpenAI streaming chunks to Anthropic streaming events."""

    yield MessageStartEvent(
        message=AnthropicMessageResponse(
            id=f"msg_{uuid.uuid4().hex[:24]}",
            content=[],
            model=request_model,
            stop_reason=None,
            usage=AnthropicUsage(input_tokens=0, output_tokens=0),
        ),
    )
    yield PingEvent()

    content_block_index = 0
    in_text_block = False
    in_tool_blocks: dict[int, bool] = {}
    tool_call_index_to_block_index: dict[int, int] = {}
    output_tokens = 0
    input_tokens = 0
    cache_read_tokens: int | None = None
    stop_reason = "end_turn"

    try:
        async for chunk in openai_stream:
            if not chunk.choices:
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
                        if in_text_block:
                            yield ContentBlockStopEvent(index=content_block_index)
                            yield PingEvent()
                            content_block_index += 1
                            in_text_block = False

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
    except Exception:
        logger.exception("Failed to stream translation response")
        yield ErrorStreamEvent(
            error=_AnthropicErrorDetail(type="api_error", message="Internal server error"),
        )
        return

    # Close any open blocks
    if in_text_block:
        yield ContentBlockStopEvent(index=content_block_index)
        yield PingEvent()

    for _tc_idx, block_idx in tool_call_index_to_block_index.items():
        yield ContentBlockStopEvent(index=block_idx)
        yield PingEvent()

    yield MessageDeltaEvent(
        delta=_MessageDelta(stop_reason=stop_reason),
        usage=AnthropicUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_input_tokens=cache_read_tokens,
        ),
    )
    yield MessageStopEvent()


def parse_anthropic_sse_event(event_type: str, data: dict[str, Any]) -> AnthropicStreamEvent | None:
    """Parse an Anthropic SSE event from its type and data."""
    if event_type == "message_start":
        return MessageStartEvent(message=AnthropicMessageResponse(**data["message"]))
    if event_type == "content_block_start":
        block_data = data["content_block"]
        content_block: (
            AnthropicTextBlock | AnthropicToolUseBlock | AnthropicThinkingBlock | AnthropicRedactedThinkingBlock
        )
        block_type = block_data.get("type")
        if block_type == "tool_use":
            content_block = AnthropicToolUseBlock(**block_data)
        elif block_type == "thinking":
            content_block = AnthropicThinkingBlock(**block_data)
        elif block_type == "redacted_thinking":
            content_block = AnthropicRedactedThinkingBlock(**block_data)
        else:
            content_block = AnthropicTextBlock(**block_data)
        return ContentBlockStartEvent(index=data["index"], content_block=content_block)
    if event_type == "content_block_delta":
        delta_data = data["delta"]
        delta_type = delta_data.get("type")
        delta: _TextDelta | _InputJsonDelta | _ThinkingDelta | _SignatureDelta
        if delta_type == "text_delta":
            delta = _TextDelta(text=delta_data["text"])
        elif delta_type == "input_json_delta":
            delta = _InputJsonDelta(partial_json=delta_data["partial_json"])
        elif delta_type == "thinking_delta":
            delta = _ThinkingDelta(thinking=delta_data["thinking"])
        elif delta_type == "signature_delta":
            delta = _SignatureDelta(signature=delta_data["signature"])
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
    if event_type == "ping":
        return PingEvent()
    if event_type == "error":
        return ErrorStreamEvent(error=_AnthropicErrorDetail(**data["error"]))
    return None
