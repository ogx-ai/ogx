# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Pydantic models for the Google Interactions API.

These models define the request and response shapes for the /v1alpha/interactions endpoint,
following the Google Interactions API specification.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

# -- Content items --


class GoogleTextContent(BaseModel):
    """A text content item."""

    type: Literal["text"] = "text"
    text: str


class GoogleFunctionCallContent(BaseModel):
    """A function call content item (model requesting a tool invocation)."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    type: Literal["function_call"] = "function_call"
    id: str | None = Field(default=None, description="Unique identifier for this function call.")
    name: str = Field(..., description="Name of the function to call.")
    args: dict[str, Any] = Field(
        default_factory=dict,
        alias="arguments",
        description="Arguments for the function call.",
    )


class GoogleFunctionResponseContent(BaseModel):
    """A function response/result content item (user providing tool results).

    Accepts both 'function_response' (generateContent API) and 'function_result'
    (Interactions API) type values, and both 'response' and 'result' field names.
    """

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    type: Literal["function_response", "function_result"] = "function_result"
    call_id: str | None = Field(default=None, description="ID of the function call this responds to.")
    id: str | None = Field(default=None, description="Deprecated alias for call_id.")
    name: str = Field(..., description="Name of the function that was called.")
    response: dict[str, Any] = Field(
        default_factory=dict,
        alias="result",
        description="The function's return value.",
    )


GoogleContentItem = GoogleTextContent | GoogleFunctionCallContent | GoogleFunctionResponseContent


# -- Conversation turns --


class GoogleInputTurn(BaseModel):
    """A conversation turn in the input."""

    role: Literal["user", "model"]
    content: str | list[GoogleContentItem] = Field(
        ...,
        description="Content items for this turn. Can be a plain string or a list of content items.",
    )

    @field_validator("content", mode="before")
    @classmethod
    def _coerce_content(cls, v: Any) -> Any:
        if isinstance(v, str):
            return v
        if isinstance(v, list):
            result = []
            for item in v:
                if isinstance(item, dict) and "type" not in item:
                    # Default to text content when type is missing
                    item = {**item, "type": "text"}
                result.append(item)
            return result
        return v


GoogleInputItem = Annotated[
    GoogleInputTurn,
    Field(description="A conversation turn."),
]


# -- Tool definitions --


class GoogleFunctionDeclaration(BaseModel):
    """A function declaration for tool calling."""

    model_config = ConfigDict(extra="allow")

    name: str = Field(..., description="The name of the function.")
    description: str | None = Field(default=None, description="Description of the function.")
    parameters: dict[str, Any] | None = Field(
        default=None,
        description="JSON Schema for the function's input parameters.",
    )


class GoogleTool(BaseModel):
    """A tool containing function declarations."""

    model_config = ConfigDict(extra="allow")

    type: str = Field(default="function", description="Tool type.")
    function_declarations: list[GoogleFunctionDeclaration] = Field(
        ...,
        description="List of function declarations.",
    )


# -- Generation config --


class GoogleGenerationConfig(BaseModel):
    """Generation parameters for the Interactions API."""

    temperature: float | None = Field(default=None, ge=0.0, le=2.0, description="Sampling temperature.")
    top_k: int | None = Field(default=None, ge=1, description="Top-k sampling parameter.")
    top_p: float | None = Field(default=None, ge=0.0, le=1.0, description="Nucleus sampling parameter.")
    max_output_tokens: int | None = Field(default=None, ge=1, description="Maximum number of tokens to generate.")


# -- Request models --


class GoogleCreateInteractionRequest(BaseModel):
    """Request body for POST /v1alpha/interactions."""

    model_config = ConfigDict(extra="allow")

    model: str = Field(..., description="The model to use for generation.")
    input: str | list[GoogleInputItem] = Field(
        ...,
        description="Prompt string or list of conversation turns.",
    )
    system_instruction: str | None = Field(
        default=None,
        description="System prompt.",
    )
    generation_config: GoogleGenerationConfig | None = Field(
        default=None,
        description="Generation parameters.",
    )
    tools: list[GoogleTool] | None = Field(
        default=None,
        description="Tools (function declarations) available to the model.",
    )
    previous_interaction_id: str | None = Field(
        default=None,
        description="ID of a previous interaction to continue the conversation from.",
    )
    stream: bool | None = Field(default=False, description="Whether to stream the response via SSE.")
    response_modalities: list[str] | None = Field(
        default=None,
        description="Accepted response modalities (e.g. ['TEXT']). Accepted for compatibility, ignored in v1.",
    )


# -- Response models --


class GoogleTextOutput(BaseModel):
    """A text output item."""

    type: Literal["text"] = "text"
    text: str


class GoogleFunctionCallOutput(BaseModel):
    """A function call output item (model requesting a tool invocation)."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    type: Literal["function_call"] = "function_call"
    id: str | None = Field(default=None, description="Unique identifier for this function call.")
    name: str = Field(..., description="Name of the function to call.")
    args: dict[str, Any] = Field(
        default_factory=dict,
        alias="arguments",
        description="Arguments for the function call.",
    )


class GoogleThoughtOutput(BaseModel):
    """A thought/reasoning output item (model's internal reasoning)."""

    model_config = ConfigDict(extra="allow")

    type: Literal["thought"] = "thought"
    signature: str | None = Field(default=None, description="Signature for the thought block.")


class GoogleUsage(BaseModel):
    """Token usage statistics."""

    model_config = ConfigDict(extra="allow")

    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_tokens: int = 0


GoogleOutputItem = Annotated[
    GoogleTextOutput | GoogleFunctionCallOutput | GoogleThoughtOutput,
    Field(discriminator="type"),
]


class GoogleInteractionResponse(BaseModel):
    """Response from POST /v1alpha/interactions (non-streaming)."""

    id: str = Field(..., description="Unique interaction ID.")
    created: str | None = Field(default=None, description="Creation timestamp.")
    status: Literal["completed"] = "completed"
    updated: str | None = Field(default=None, description="Last update timestamp.")
    model: str = Field(..., description="Model used for generation.")
    outputs: list[GoogleOutputItem] = Field(..., description="Response output items.")
    role: Literal["model"] = "model"
    usage: GoogleUsage = Field(default_factory=GoogleUsage)
    object: Literal["interaction"] = "interaction"


# -- Streaming event models --


class _InteractionRef(BaseModel):
    """Interaction reference used in streaming events."""

    id: str
    status: str = "in_progress"
    model: str | None = None
    object: Literal["interaction"] = "interaction"


class _InteractionCompleteRef(BaseModel):
    """Full interaction reference used in the complete event."""

    id: str
    created: str | None = None
    status: Literal["completed"] = "completed"
    updated: str | None = None
    model: str | None = None
    role: Literal["model"] = "model"
    usage: GoogleUsage = Field(default_factory=GoogleUsage)
    object: Literal["interaction"] = "interaction"


class InteractionStartEvent(BaseModel):
    """First event in a streaming response."""

    event_type: Literal["interaction.start"] = "interaction.start"
    interaction: _InteractionRef


class _ContentRef(BaseModel):
    """Content type reference used in content.start events."""

    type: Literal["text", "function_call"] = "text"


class _FunctionCallContentRef(BaseModel):
    """Content reference for function_call content.start events."""

    type: Literal["function_call"] = "function_call"
    id: str | None = None
    name: str | None = None


class ContentStartEvent(BaseModel):
    """Signals the start of a new content block."""

    event_type: Literal["content.start"] = "content.start"
    index: int
    content: _ContentRef | _FunctionCallContentRef = Field(default_factory=_ContentRef)


class _TextDelta(BaseModel):
    type: Literal["text"] = "text"
    text: str


class _FunctionCallDelta(BaseModel):
    type: Literal["function_call"] = "function_call"
    args: str = Field(..., description="Partial JSON string for function call arguments.")


class ContentDeltaEvent(BaseModel):
    """A delta within a content block."""

    event_type: Literal["content.delta"] = "content.delta"
    index: int
    delta: _TextDelta | _FunctionCallDelta


class ContentStopEvent(BaseModel):
    """Signals the end of a content block."""

    event_type: Literal["content.stop"] = "content.stop"
    index: int


class InteractionCompleteEvent(BaseModel):
    """Final event in a streaming response."""

    event_type: Literal["interaction.complete"] = "interaction.complete"
    interaction: _InteractionCompleteRef


GoogleStreamEvent = (
    InteractionStartEvent | ContentStartEvent | ContentDeltaEvent | ContentStopEvent | InteractionCompleteEvent
)


# -- Error response --


class _GoogleErrorDetail(BaseModel):
    code: int
    message: str


class GoogleErrorResponse(BaseModel):
    """Google-format error response."""

    error: _GoogleErrorDetail
