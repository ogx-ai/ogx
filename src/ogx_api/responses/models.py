# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Pydantic models for Responses API requests and responses.

This module defines the request and response models for the Responses API
using Pydantic with Field descriptions for OpenAPI schema generation.
"""

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from ogx_api.common.responses import Order
from ogx_api.inference import ServiceTier
from ogx_api.openai_responses import (
    OpenAIResponseInput,
    OpenAIResponseInputTool,
    OpenAIResponseInputToolChoice,
    OpenAIResponsePrompt,
    OpenAIResponseReasoning,
    OpenAIResponseText,
)
from ogx_api.schema_utils import remove_null_from_anyof


class ResponseItemInclude(StrEnum):
    """Specify additional output data to include in the model response."""

    web_search_call_action_sources = "web_search_call.action.sources"
    code_interpreter_call_outputs = "code_interpreter_call.outputs"
    computer_call_output_output_image_url = "computer_call_output.output.image_url"
    file_search_call_results = "file_search_call.results"
    message_input_image_image_url = "message.input_image.image_url"
    message_output_text_logprobs = "message.output_text.logprobs"
    reasoning_encrypted_content = "reasoning.encrypted_content"


class ResponseTruncation(StrEnum):
    """Controls how the service truncates input when it exceeds the model context window."""

    auto = "auto"  # Let the service decide how to truncate
    disabled = "disabled"  # Disable truncation; context over limit results in 400 error


class ResponseGuardrailSpec(BaseModel):
    """Specification for a guardrail to apply during response generation."""

    model_config = ConfigDict(extra="forbid")

    type: str
    # TODO: more fields to be added for guardrail configuration


ResponseGuardrail = str | ResponseGuardrailSpec


class ResponseStreamOptions(BaseModel):
    """Options that control streamed response behavior."""

    model_config = ConfigDict(extra="forbid")

    include_obfuscation: bool = Field(
        default=True,
        description="Whether to obfuscate sensitive information in streamed output.",
    )


class ContextManagement(BaseModel):
    """Configuration for automatic context management during response generation."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["compaction"] = Field(
        ..., description="The context management entry type. Currently only 'compaction' is supported."
    )
    compact_threshold: int | None = Field(
        default=None, description="Token threshold at which compaction should be triggered."
    )


# extra_body can be accessed via .model_extra
class CreateResponseRequest(BaseModel):
    """Request model for creating a response."""

    model_config = ConfigDict(extra="allow")

    input: str | list[OpenAIResponseInput] = Field(..., description="Input message(s) to create the response.")
    model: str = Field(..., description="The underlying LLM used for completions.")
    background: bool | None = Field(
        default=None,
        description="Whether to run the model response in the background. When true, returns immediately with status 'queued'.",
        json_schema_extra=remove_null_from_anyof,
    )
    prompt: OpenAIResponsePrompt | None = Field(
        default=None, description="Prompt object with ID, version, and variables."
    )
    instructions: str | None = Field(default=None, description="Instructions to guide the model's behavior.")
    parallel_tool_calls: bool | None = Field(
        default=True,
        description="Whether to enable parallel tool calls.",
    )
    previous_response_id: str | None = Field(
        default=None,
        description="Optional ID of a previous response to continue from.",
    )
    prompt_cache_key: str | None = Field(
        default=None,
        max_length=64,
        description="A key to use when reading from or writing to the prompt cache.",
    )
    conversation: str | None = Field(
        default=None,
        description="Optional ID of a conversation to add the response to.",
    )
    store: bool | None = Field(
        default=True,
        description="Whether to store the response in the database.",
        json_schema_extra=remove_null_from_anyof,
    )
    stream: bool | None = Field(
        default=False,
        description="Whether to stream the response.",
        json_schema_extra=remove_null_from_anyof,
    )
    temperature: float | None = Field(
        default=None,
        ge=0.0,
        le=2.0,
        description="Sampling temperature.",
    )
    top_p: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Nucleus sampling parameter that controls response diversity (lower values increase focus).",
    )
    frequency_penalty: float | None = Field(
        default=None,
        ge=-2.0,
        le=2.0,
        description="Penalizes new tokens based on their frequency in the text so far.",
    )
    text: OpenAIResponseText | None = Field(
        default=None,
        description="Configuration for text response generation.",
    )
    tool_choice: OpenAIResponseInputToolChoice | None = Field(
        default=None,
        description="How the model should select which tool to call (if any).",
    )
    tools: list[OpenAIResponseInputTool] | None = Field(
        default=None,
        description="List of tools available to the model.",
    )
    include: list[ResponseItemInclude] | None = Field(
        default=None,
        description="Additional fields to include in the response.",
        json_schema_extra=remove_null_from_anyof,
    )
    max_infer_iters: int | None = Field(
        default=10,
        ge=1,
        description="Maximum number of inference iterations.",
    )
    guardrails: list[ResponseGuardrail] | None = Field(
        default=None,
        description="List of guardrails to apply during response generation.",
    )
    max_tool_calls: int | None = Field(
        default=None,
        ge=1,
        description="Max number of total calls to built-in tools that can be processed in a response.",
    )
    max_output_tokens: int | None = Field(
        default=None,
        ge=16,
        description="Upper bound for the number of tokens that can be generated for a response.",
    )
    reasoning: OpenAIResponseReasoning | None = Field(
        default=None,
        description="Configuration for reasoning effort in responses.",
    )
    safety_identifier: str | None = Field(
        default=None,
        max_length=64,
        description="A stable identifier used for safety monitoring and abuse detection.",
    )
    service_tier: ServiceTier | None = Field(
        default=None,
        description="The service tier to use for this request.",
    )
    metadata: dict[str, str] | None = Field(
        default=None,
        description="Dictionary of metadata key-value pairs to attach to the response.",
    )
    truncation: ResponseTruncation | None = Field(
        default=None,
        description="Controls how the service truncates input when it exceeds the model context window.",
    )
    top_logprobs: int | None = Field(
        default=None,
        ge=0,
        le=20,
        description="The number of most likely tokens to return at each position, along with their log probabilities.",
    )
    presence_penalty: float | None = Field(
        default=None,
        ge=-2.0,
        le=2.0,
        description="Penalizes new tokens based on whether they appear in the text so far.",
    )
    stream_options: ResponseStreamOptions | None = Field(
        default=None,
        description="Options that control streamed response behavior.",
    )
    context_management: list[ContextManagement] | None = Field(
        default=None,
        description="Context management configuration. When set with type 'compaction', automatically compacts conversation history when token count exceeds the compact_threshold.",
    )


class RetrieveResponseRequest(BaseModel):
    """Request model for retrieving a response."""

    model_config = ConfigDict(extra="forbid")

    response_id: str = Field(..., min_length=1, description="The ID of the OpenAI response to retrieve.")


class ListResponsesRequest(BaseModel):
    """Request model for listing responses."""

    model_config = ConfigDict(extra="forbid")

    after: str | None = Field(default=None, description="The ID of the last response to return.")
    limit: int | None = Field(default=50, ge=1, le=100, description="The number of responses to return.")
    model: str | None = Field(default=None, description="The model to filter responses by.")
    order: Order | None = Field(
        default=Order.desc,
        description="The order to sort responses by when sorted by created_at ('asc' or 'desc').",
    )


class ListResponseInputItemsRequest(BaseModel):
    """Request model for listing input items of a response."""

    model_config = ConfigDict(extra="forbid")

    response_id: str = Field(..., min_length=1, description="The ID of the response to retrieve input items for.")
    after: str | None = Field(default=None, description="An item ID to list items after, used for pagination.")
    before: str | None = Field(default=None, description="An item ID to list items before, used for pagination.")
    include: list[ResponseItemInclude] | None = Field(
        default=None, description="Additional fields to include in the response."
    )
    limit: int | None = Field(
        default=20,
        ge=1,
        le=100,
        description="A limit on the number of objects to be returned. Limit can range between 1 and 100, and the default is 20.",
    )
    order: Order | None = Field(default=Order.desc, description="The order to return the input items in.")


class CompactResponseRequest(BaseModel):
    """Request model for compacting a conversation."""

    model_config = ConfigDict(extra="allow")

    model: str = Field(..., description="The model to use for generating the compacted summary.")
    input: str | list[OpenAIResponseInput] | None = Field(default=None, description="Input message(s) to compact.")
    instructions: str | None = Field(default=None, description="Instructions to guide the compaction.")
    previous_response_id: str | None = Field(
        default=None, description="ID of a previous response whose history to compact."
    )
    prompt_cache_key: str | None = Field(
        default=None,
        max_length=64,
        description="A key to use when reading from or writing to the prompt cache.",
    )
    tools: list[OpenAIResponseInputTool] | None = Field(
        default=None,
        description="List of tools available to the model. Accepted for compatibility but not used during compaction.",
    )
    parallel_tool_calls: bool | None = Field(
        default=None,
        description="Whether to enable parallel tool calls. Accepted for compatibility but not used during compaction.",
    )
    reasoning: OpenAIResponseReasoning | None = Field(
        default=None,
        description="Configuration for reasoning effort. Accepted for compatibility but not used during compaction.",
    )
    text: OpenAIResponseText | None = Field(
        default=None,
        description="Configuration for text response generation. Accepted for compatibility but not used during compaction.",
    )


class DeleteResponseRequest(BaseModel):
    """Request model for deleting a response."""

    model_config = ConfigDict(extra="forbid")

    response_id: str = Field(..., min_length=1, description="The ID of the OpenAI response to delete.")


class CancelResponseRequest(BaseModel):
    """Request model for canceling a response."""

    model_config = ConfigDict(extra="forbid")

    response_id: str = Field(..., min_length=1, description="The ID of the OpenAI response to cancel.")
