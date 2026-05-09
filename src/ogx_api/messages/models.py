# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Pydantic models for the Anthropic Messages API.

These models define the request and response shapes for the /v1/messages endpoint,
following the Anthropic Messages API specification.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

# Anthropic API version we are compatible with
ANTHROPIC_VERSION = "2023-06-01"

# -- Content blocks --


class AnthropicTextBlock(BaseModel):
    """A text content block."""

    type: Literal["text"] = "text"
    text: str


class AnthropicImageSource(BaseModel):
    """Source for an image content block."""

    type: Literal["base64"] = "base64"
    media_type: str = Field(..., description="MIME type of the image (e.g. image/png).")
    data: str = Field(..., description="Base64-encoded image data.")


class AnthropicImageBlock(BaseModel):
    """An image content block."""

    type: Literal["image"] = "image"
    source: AnthropicImageSource


class AnthropicToolUseBlock(BaseModel):
    """A tool use content block in an assistant message."""

    type: Literal["tool_use"] = "tool_use"
    id: str = Field(..., description="Unique ID for this tool invocation.")
    name: str = Field(..., description="Name of the tool being called.")
    input: dict[str, Any] = Field(..., description="Tool input arguments.")


class AnthropicToolResultBlock(BaseModel):
    """A tool result content block in a user message."""

    type: Literal["tool_result"] = "tool_result"
    tool_use_id: str = Field(..., description="The ID of the tool_use block this result corresponds to.")
    content: str | list[AnthropicTextBlock | AnthropicImageBlock] = Field(
        default="",
        description="The result content.",
    )
    is_error: bool | None = Field(default=None, description="Whether the tool call resulted in an error.")


class AnthropicThinkingBlock(BaseModel):
    """A thinking content block (extended thinking)."""

    type: Literal["thinking"] = "thinking"
    thinking: str = Field(..., description="The model's thinking text.")
    signature: str | None = Field(default=None, description="Signature for the thinking block.")


AnthropicContentBlock = Annotated[
    AnthropicTextBlock
    | AnthropicImageBlock
    | AnthropicToolUseBlock
    | AnthropicToolResultBlock
    | AnthropicThinkingBlock,
    Field(discriminator="type"),
]

# -- Messages --


class AnthropicMessage(BaseModel):
    """A message in the conversation."""

    role: Literal["user", "assistant"]
    content: str | list[AnthropicContentBlock] = Field(
        ...,
        description="Message content: a string for simple text, or a list of content blocks.",
    )


# -- Tool definitions --


class AnthropicToolDef(BaseModel):
    """Definition of a tool available to the model."""

    name: str
    description: str | None = None
    input_schema: dict[str, Any] = Field(..., description="JSON Schema for the tool's input.")


# -- Thinking config --


class AnthropicThinkingConfig(BaseModel):
    """Configuration for extended thinking."""

    type: Literal["enabled", "disabled", "adaptive"] = "enabled"
    budget_tokens: int | None = Field(default=None, ge=1, description="Maximum tokens for thinking.")


# -- Request models --


class AnthropicCreateMessageRequest(BaseModel):
    """Request body for POST /v1/messages."""

    model_config = ConfigDict(extra="allow")

    model: str = Field(..., description="The model to use for generation.")
    messages: list[AnthropicMessage] = Field(..., description="The messages in the conversation.")
    max_tokens: int = Field(..., ge=1, description="The maximum number of tokens to generate.")
    system: str | list[AnthropicTextBlock] | None = Field(
        default=None,
        description="System prompt. A string or list of text blocks.",
    )
    tools: list[AnthropicToolDef] | None = Field(default=None, description="Tools available to the model.")
    tool_choice: Any | None = Field(
        default=None,
        description="How the model should select tools. One of: 'auto', 'any', 'none', or {type: 'tool', name: '...'}.",
    )
    stream: bool | None = Field(default=False, description="Whether to stream the response.")
    temperature: float | None = Field(default=None, ge=0.0, le=1.0, description="Sampling temperature.")
    top_p: float | None = Field(default=None, ge=0.0, le=1.0, description="Nucleus sampling parameter.")
    top_k: int | None = Field(default=None, ge=1, description="Top-k sampling parameter.")
    stop_sequences: list[str] | None = Field(default=None, description="Custom stop sequences.")
    metadata: dict[str, str] | None = Field(default=None, description="Request metadata.")
    thinking: AnthropicThinkingConfig | None = Field(default=None, description="Extended thinking configuration.")
    service_tier: str | None = Field(default=None, description="Service tier to use.")


class AnthropicCountTokensRequest(BaseModel):
    """Request body for POST /v1/messages/count_tokens."""

    model: str = Field(..., description="The model to use for token counting.")
    messages: list[AnthropicMessage] = Field(..., description="The messages to count tokens for.")
    system: str | list[AnthropicTextBlock] | None = Field(default=None, description="System prompt.")
    tools: list[AnthropicToolDef] | None = Field(default=None, description="Tools to include in token count.")


# -- Response models --


class AnthropicUsage(BaseModel):
    """Token usage statistics."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int | None = None
    cache_read_input_tokens: int | None = None


class AnthropicMessageResponse(BaseModel):
    """Response from POST /v1/messages (non-streaming)."""

    id: str = Field(..., description="Unique message ID (msg_ prefix).")
    type: Literal["message"] = "message"
    role: Literal["assistant"] = "assistant"
    content: list[AnthropicContentBlock] = Field(..., description="Response content blocks.")
    model: str
    stop_reason: str | None = Field(
        default=None,
        description="Why the model stopped: end_turn, stop_sequence, tool_use, or max_tokens.",
    )
    stop_sequence: str | None = None
    usage: AnthropicUsage = Field(default_factory=AnthropicUsage)


class AnthropicCountTokensResponse(BaseModel):
    """Response from POST /v1/messages/count_tokens."""

    input_tokens: int


# -- Streaming event models --


class MessageStartEvent(BaseModel):
    """First event in a streaming response."""

    type: Literal["message_start"] = "message_start"
    message: AnthropicMessageResponse


class ContentBlockStartEvent(BaseModel):
    """Signals the start of a new content block."""

    type: Literal["content_block_start"] = "content_block_start"
    index: int
    content_block: AnthropicContentBlock


class _TextDelta(BaseModel):
    type: Literal["text_delta"] = "text_delta"
    text: str


class _InputJsonDelta(BaseModel):
    type: Literal["input_json_delta"] = "input_json_delta"
    partial_json: str


class _ThinkingDelta(BaseModel):
    type: Literal["thinking_delta"] = "thinking_delta"
    thinking: str


class ContentBlockDeltaEvent(BaseModel):
    """A delta within a content block."""

    type: Literal["content_block_delta"] = "content_block_delta"
    index: int
    delta: _TextDelta | _InputJsonDelta | _ThinkingDelta


class ContentBlockStopEvent(BaseModel):
    """Signals the end of a content block."""

    type: Literal["content_block_stop"] = "content_block_stop"
    index: int


class _MessageDelta(BaseModel):
    stop_reason: str | None = None
    stop_sequence: str | None = None


class MessageDeltaEvent(BaseModel):
    """Final metadata update before the message ends."""

    type: Literal["message_delta"] = "message_delta"
    delta: _MessageDelta
    usage: AnthropicUsage | None = None


class MessageStopEvent(BaseModel):
    """Final event in a streaming response."""

    type: Literal["message_stop"] = "message_stop"


AnthropicStreamEvent = (
    MessageStartEvent
    | ContentBlockStartEvent
    | ContentBlockDeltaEvent
    | ContentBlockStopEvent
    | MessageDeltaEvent
    | MessageStopEvent
)


# -- Error response --


class _AnthropicErrorDetail(BaseModel):
    type: str
    message: str


class AnthropicErrorResponse(BaseModel):
    """Anthropic-format error response."""

    type: Literal["error"] = "error"
    error: _AnthropicErrorDetail
