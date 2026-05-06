# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Pydantic models for Inference API requests and responses.

This module defines all request and response models for the Inference API
using Pydantic with Field descriptions for OpenAPI schema generation.
"""

from enum import Enum, StrEnum
from typing import Annotated, Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from typing_extensions import TypedDict

from ogx_api.common.content_types import InterleavedContent
from ogx_api.common.responses import Order
from ogx_api.schema_utils import (
    json_schema_type,
    nullable_openai_style,
    register_schema,
    remove_null_from_anyof,
)


# Sampling Strategies
@json_schema_type
class GreedySamplingStrategy(BaseModel):
    """Greedy sampling strategy that selects the highest probability token at each step."""

    type: Literal["greedy"] = Field(
        default="greedy", description="Must be 'greedy' to identify this sampling strategy."
    )


@json_schema_type
class TopPSamplingStrategy(BaseModel):
    """Top-p (nucleus) sampling strategy that samples from the smallest set of tokens with cumulative probability >= p."""

    type: Literal["top_p"] = Field(default="top_p", description="Must be 'top_p' to identify this sampling strategy.")
    temperature: float = Field(
        ..., gt=0.0, le=2.0, description="Controls randomness in sampling. Higher values increase randomness."
    )
    top_p: float = Field(
        default=0.95, ge=0.0, le=1.0, description="Cumulative probability threshold for nucleus sampling."
    )


@json_schema_type
class TopKSamplingStrategy(BaseModel):
    """Top-k sampling strategy that restricts sampling to the k most likely tokens."""

    type: Literal["top_k"] = Field(default="top_k", description="Must be 'top_k' to identify this sampling strategy.")
    top_k: int = Field(..., ge=1, description="Number of top tokens to consider for sampling. Must be at least 1.")


SamplingStrategy = Annotated[
    GreedySamplingStrategy | TopPSamplingStrategy | TopKSamplingStrategy,
    Field(discriminator="type"),
]
register_schema(SamplingStrategy, name="SamplingStrategy")


@json_schema_type
class SamplingParams(BaseModel):
    """Sampling parameters for text generation."""

    strategy: SamplingStrategy = Field(
        default_factory=GreedySamplingStrategy, description="The sampling strategy to use."
    )
    max_tokens: int | None = Field(
        default=None,
        ge=1,
        description="The maximum number of tokens that can be generated in the completion. The token count of your prompt plus max_tokens cannot exceed the model's context length.",
    )
    repetition_penalty: float | None = Field(
        default=1.0,
        ge=-2.0,
        le=2.0,
        description="Number between -2.0 and 2.0. Positive values penalize new tokens based on whether they appear in the text so far.",
    )
    stop: list[str] | None = Field(
        default=None,
        max_length=4,
        description="Up to 4 sequences where the API will stop generating further tokens. The returned text will not contain the stop sequence.",
    )


class LogProbConfig(BaseModel):
    """Configuration for log probability output."""

    top_k: int | None = Field(
        default=0, ge=0, description="How many tokens (for each position) to return log probabilities for."
    )


class QuantizationType(Enum):
    """Type of model quantization to run inference with."""

    bf16 = "bf16"
    fp8_mixed = "fp8_mixed"
    int4_mixed = "int4_mixed"


@json_schema_type
class Fp8QuantizationConfig(BaseModel):
    """Configuration for 8-bit floating point quantization."""

    type: Literal["fp8_mixed"] = Field(
        default="fp8_mixed", description="Must be 'fp8_mixed' to identify this quantization type."
    )


@json_schema_type
class Bf16QuantizationConfig(BaseModel):
    """Configuration for BFloat16 precision (typically no quantization)."""

    type: Literal["bf16"] = Field(default="bf16", description="Must be 'bf16' to identify this quantization type.")


@json_schema_type
class Int4QuantizationConfig(BaseModel):
    """Configuration for 4-bit integer quantization."""

    type: Literal["int4_mixed"] = Field(
        default="int4_mixed", description="Must be 'int4' to identify this quantization type."
    )
    scheme: str | None = Field(default="int4_weight_int8_dynamic_activation", description="Quantization scheme to use.")


QuantizationConfig = Annotated[
    Bf16QuantizationConfig | Fp8QuantizationConfig | Int4QuantizationConfig,
    Field(discriminator="type"),
]


# Message Models
@json_schema_type
class UserMessage(BaseModel):
    """A message from the user in a chat conversation."""

    role: Literal["user"] = Field(default="user", description="Must be 'user' to identify this as a user message.")
    content: InterleavedContent = Field(
        ..., description="The content of the message, which can include text and other media."
    )
    context: InterleavedContent | None = Field(
        default=None,
        description="This field is used internally by OGX to pass RAG context. This field may be removed in the API in the future.",
    )


@json_schema_type
class SystemMessage(BaseModel):
    """A system message providing instructions or context to the model."""

    role: Literal["system"] = Field(
        default="system", description="Must be 'system' to identify this as a system message."
    )
    content: InterleavedContent = Field(
        ...,
        description="The content of the 'system prompt'. If multiple system messages are provided, they are concatenated. The underlying OGX code may also add other system messages.",
    )


@json_schema_type
class ToolResponseMessage(BaseModel):
    """A message representing the result of a tool invocation."""

    role: Literal["tool"] = Field(default="tool", description="Must be 'tool' to identify this as a tool response.")
    call_id: str = Field(..., description="Unique identifier for the tool call this response is for.")
    content: InterleavedContent = Field(..., description="The response content from the tool.")


class ToolChoice(Enum):
    """Whether tool use is required or automatic. This is a hint to the model which may not be followed."""

    auto = "auto"
    required = "required"
    none = "none"


@json_schema_type
class TokenLogProbs(BaseModel):
    """Log probabilities for generated tokens."""

    logprobs_by_token: dict[str, float] = Field(
        ..., description="Dictionary mapping tokens to their log probabilities."
    )


class ChatCompletionResponseEventType(Enum):
    """Types of events that can occur during chat completion."""

    start = "start"
    complete = "complete"
    progress = "progress"


class ResponseFormatType(StrEnum):
    """Types of formats for structured (guided) decoding."""

    json_schema = "json_schema"
    grammar = "grammar"


@json_schema_type
class JsonSchemaResponseFormat(BaseModel):
    """Configuration for JSON schema-guided response generation."""

    type: Literal[ResponseFormatType.json_schema] = Field(
        default=ResponseFormatType.json_schema, description="Must be 'json_schema' to identify this format type."
    )
    json_schema: dict[str, Any] = Field(..., description="The JSON schema the response should conform to.")


@json_schema_type
class GrammarResponseFormat(BaseModel):
    """Configuration for grammar-guided response generation."""

    type: Literal[ResponseFormatType.grammar] = Field(
        default=ResponseFormatType.grammar, description="Must be 'grammar' to identify this format type."
    )
    bnf: dict[str, Any] = Field(..., description="The BNF grammar specification the response should conform to.")


ResponseFormat = Annotated[
    JsonSchemaResponseFormat | GrammarResponseFormat,
    Field(discriminator="type"),
]
register_schema(ResponseFormat, name="ResponseFormat")


# This is an internally used class
class CompletionRequest(BaseModel):
    """Internal request model for text completion inference."""

    model: str
    content: InterleavedContent
    sampling_params: SamplingParams | None = Field(default_factory=SamplingParams)
    response_format: ResponseFormat | None = None
    stream: bool | None = False
    logprobs: LogProbConfig | None = None


class SystemMessageBehavior(Enum):
    """Config for how to override the default system prompt."""

    append = "append"
    replace = "replace"


@json_schema_type
class EmbeddingsResponse(BaseModel):
    """Response containing generated embeddings."""

    embeddings: list[list[float]] = Field(
        ...,
        description="List of embedding vectors, one per input content. Each embedding is a list of floats. The dimensionality is model-specific.",
    )


@json_schema_type
class RerankData(BaseModel):
    """A single rerank result from a reranking response."""

    index: int = Field(..., ge=0, description="The original index of the document in the input list.")
    relevance_score: float = Field(
        ..., description="The relevance score from the model output. Higher scores indicate greater relevance."
    )


@json_schema_type
class RerankResponse(BaseModel):
    """Response from a reranking request."""

    data: list[RerankData] = Field(
        ..., description="List of rerank result objects, sorted by relevance score (descending)."
    )


# OpenAI Compatibility Models
@json_schema_type
class OpenAIChatCompletionContentPartTextParam(BaseModel):
    """Text content part for OpenAI-compatible chat completion messages."""

    type: Literal["text"] = Field(default="text", description="Must be 'text' to identify this as text content.")
    text: str = Field(..., description="The text content of the message.")


@json_schema_type
class OpenAIImageURL(BaseModel):
    """Image URL specification for OpenAI-compatible chat completion messages."""

    url: str = Field(..., description="URL of the image to include in the message.")
    detail: Literal["low", "high", "auto"] | None = Field(
        default=None, description="Level of detail for image processing. Can be 'low', 'high', or 'auto'."
    )


@json_schema_type
class OpenAIChatCompletionContentPartImageParam(BaseModel):
    """Image content part for OpenAI-compatible chat completion messages."""

    type: Literal["image_url"] = Field(
        default="image_url", description="Must be 'image_url' to identify this as image content."
    )
    image_url: OpenAIImageURL = Field(..., description="Image URL specification and processing details.")


@json_schema_type
class OpenAIFileFile(BaseModel):
    """File reference for OpenAI-compatible file content."""

    file_data: str | None = Field(default=None, description="Base64-encoded file data.")
    file_id: str | None = Field(default=None, description="ID of an uploaded file.")
    filename: str | None = Field(default=None, description="Name of the file.")

    @model_validator(mode="after")
    def validate_file_reference(self) -> Self:
        """Ensure at least one of file_data or file_id is provided."""
        if self.file_data is None and self.file_id is None:
            raise ValueError("Either file_data or file_id must be provided")
        return self


@json_schema_type
class OpenAIFile(BaseModel):
    """File content part for OpenAI-compatible chat completion messages."""

    type: Literal["file"] = Field(default="file", description="Must be 'file' to identify this as file content.")
    file: OpenAIFileFile = Field(..., description="File specification.")


OpenAIChatCompletionContentPartParam = Annotated[
    OpenAIChatCompletionContentPartTextParam | OpenAIChatCompletionContentPartImageParam | OpenAIFile,
    Field(discriminator="type"),
]
register_schema(OpenAIChatCompletionContentPartParam, name="OpenAIChatCompletionContentPartParam")


OpenAIChatCompletionMessageContent = str | list[OpenAIChatCompletionContentPartParam]

OpenAIChatCompletionTextOnlyMessageContent = str | list[OpenAIChatCompletionContentPartTextParam]


@json_schema_type
class OpenAIUserMessageParam(BaseModel):
    """A message from the user in an OpenAI-compatible chat completion request."""

    role: Literal["user"] = Field(default="user", description="Must be 'user' to identify this as a user message.")
    content: OpenAIChatCompletionMessageContent = Field(
        ..., description="The content of the message, which can include text and other media."
    )
    name: str | None = Field(default=None, description="The name of the user message participant.")


@json_schema_type
class OpenAISystemMessageParam(BaseModel):
    """A system message providing instructions or context to the model."""

    role: Literal["system"] = Field(
        default="system", description="Must be 'system' to identify this as a system message."
    )
    content: OpenAIChatCompletionTextOnlyMessageContent = Field(
        ...,
        description="The content of the 'system prompt'. If multiple system messages are provided, they are concatenated.",
    )
    name: str | None = Field(default=None, description="The name of the system message participant.")


@json_schema_type
class OpenAIChatCompletionToolCallFunction(BaseModel):
    """Function call details for OpenAI-compatible tool calls."""

    name: str = Field(..., description="Name of the function to call.")
    arguments: str = Field(..., description="Arguments to pass to the function as a JSON string.")


@json_schema_type
class OpenAIChatCompletionToolCall(BaseModel):
    """Tool call specification for OpenAI-compatible chat completion responses."""

    index: int | None = Field(default=None, ge=0, description="Index of the tool call in the list.")
    id: str | None = Field(default=None, description="Unique identifier for the tool call.")
    type: Literal["function"] = Field(
        default="function", description="Must be 'function' to identify this as a function call."
    )
    function: OpenAIChatCompletionToolCallFunction | None = Field(default=None, description="Function call details.")


class OpenAIChatCompletionCustomToolCallFunction(BaseModel):
    """Custom tool call details for OpenAI-compatible tool calls."""

    name: str = Field(..., description="The name of the custom tool to call.")
    input: str = Field(..., description="The input for the custom tool call generated by the model.")


@json_schema_type
class OpenAIChatCompletionCustomToolCall(BaseModel):
    """A call to a custom tool created by the model."""

    id: str = Field(..., description="The ID of the tool call.")
    type: Literal["custom"] = Field(default="custom", description="The type of the tool. Always 'custom'.")
    custom: OpenAIChatCompletionCustomToolCallFunction = Field(
        ..., description="The custom tool that the model called."
    )


@json_schema_type
class OpenAIAssistantMessageParam(BaseModel):
    """A message containing the model's (assistant) response in an OpenAI-compatible chat completion request."""

    model_config = ConfigDict(extra="allow")

    role: Literal["assistant"] = Field(
        default="assistant", description="Must be 'assistant' to identify this as the model's response."
    )
    content: OpenAIChatCompletionTextOnlyMessageContent | None = Field(
        default=None, description="The content of the model's response."
    )
    name: str | None = Field(default=None, description="The name of the assistant message participant.")
    tool_calls: list[OpenAIChatCompletionToolCall] | None = Field(
        default=None, description="List of tool calls. Each tool call is an OpenAIChatCompletionToolCall object."
    )


@json_schema_type
class OpenAIToolMessageParam(BaseModel):
    """A message representing the result of a tool invocation in an OpenAI-compatible chat completion request."""

    role: Literal["tool"] = Field(default="tool", description="Must be 'tool' to identify this as a tool response.")
    tool_call_id: str = Field(..., description="Unique identifier for the tool call this response is for.")
    content: OpenAIChatCompletionTextOnlyMessageContent = Field(..., description="The response content from the tool.")


@json_schema_type
class OpenAIDeveloperMessageParam(BaseModel):
    """A message from the developer in an OpenAI-compatible chat completion request."""

    role: Literal["developer"] = Field(
        default="developer", description="Must be 'developer' to identify this as a developer message."
    )
    content: OpenAIChatCompletionTextOnlyMessageContent = Field(
        ..., description="The content of the developer message."
    )
    name: str | None = Field(default=None, description="The name of the developer message participant.")


OpenAIMessageParam = Annotated[
    OpenAIUserMessageParam
    | OpenAISystemMessageParam
    | OpenAIAssistantMessageParam
    | OpenAIToolMessageParam
    | OpenAIDeveloperMessageParam,
    Field(discriminator="role"),
]
register_schema(OpenAIMessageParam, name="OpenAIMessageParam")


@json_schema_type
class OpenAIResponseFormatText(BaseModel):
    """Text response format for OpenAI-compatible chat completion requests."""

    type: Literal["text"] = Field(default="text", description="Must be 'text' to indicate plain text response format.")


@json_schema_type
class OpenAIJSONSchema(TypedDict, total=False):
    """JSON schema specification for OpenAI-compatible structured response format."""

    name: str
    description: str | None
    strict: bool | None

    # Pydantic BaseModel cannot be used with a schema param, since it already
    # has one. And, we don't want to alias here because then have to handle
    # that alias when converting to OpenAI params. So, to support schema,
    # we use a TypedDict.
    schema: dict[str, Any] | None


@json_schema_type
class OpenAIResponseFormatJSONSchema(BaseModel):
    """JSON schema response format for OpenAI-compatible chat completion requests."""

    type: Literal["json_schema"] = Field(
        default="json_schema", description="Must be 'json_schema' to indicate structured JSON response format."
    )
    json_schema: OpenAIJSONSchema = Field(..., description="The JSON schema specification for the response.")


@json_schema_type
class OpenAIResponseFormatJSONObject(BaseModel):
    """JSON object response format for OpenAI-compatible chat completion requests."""

    type: Literal["json_object"] = Field(
        default="json_object", description="Must be 'json_object' to indicate generic JSON object response format."
    )


OpenAIResponseFormatParam = Annotated[
    OpenAIResponseFormatText | OpenAIResponseFormatJSONSchema | OpenAIResponseFormatJSONObject,
    Field(discriminator="type"),
]
register_schema(OpenAIResponseFormatParam, name="OpenAIResponseFormatParam")


@json_schema_type
class FunctionToolConfig(BaseModel):
    """Configuration for a function tool specifying the function name."""

    name: str = Field(..., description="Name of the function.")


@json_schema_type
class OpenAIChatCompletionToolChoiceFunctionTool(BaseModel):
    """Function tool choice for OpenAI-compatible chat completion requests."""

    type: Literal["function"] = Field(
        default="function", description="Must be 'function' to indicate function tool choice."
    )
    function: FunctionToolConfig = Field(..., description="The function tool configuration.")

    def __init__(self, name: str):
        super().__init__(type="function", function=FunctionToolConfig(name=name))


@json_schema_type
class CustomToolConfig(BaseModel):
    """Custom tool configuration for OpenAI-compatible chat completion requests."""

    name: str = Field(..., description="Name of the custom tool.")


@json_schema_type
class OpenAIChatCompletionToolChoiceCustomTool(BaseModel):
    """Custom tool choice for OpenAI-compatible chat completion requests."""

    type: Literal["custom"] = Field(default="custom", description="Must be 'custom' to indicate custom tool choice.")
    custom: CustomToolConfig = Field(..., description="Custom tool configuration.")

    def __init__(self, name: str):
        super().__init__(type="custom", custom=CustomToolConfig(name=name))


@json_schema_type
class AllowedToolsConfig(BaseModel):
    """Configuration specifying which tools are allowed and their selection mode."""

    tools: list[dict[str, Any]] = Field(..., description="List of allowed tools.")
    mode: Literal["auto", "required"] = Field(..., description="Mode for allowed tools.")


@json_schema_type
class OpenAIChatCompletionToolChoiceAllowedTools(BaseModel):
    """Allowed tools response format for OpenAI-compatible chat completion requests."""

    type: Literal["allowed_tools"] = Field(
        default="allowed_tools", description="Must be 'allowed_tools' to indicate allowed tools response format."
    )
    allowed_tools: AllowedToolsConfig = Field(..., description="Allowed tools configuration.")

    def __init__(self, tools: list[dict[str, Any]], mode: Literal["auto", "required"]):
        super().__init__(type="allowed_tools", allowed_tools=AllowedToolsConfig(tools=tools, mode=mode))


# Define the object-level union with discriminator
OpenAIChatCompletionToolChoice = Annotated[
    OpenAIChatCompletionToolChoiceAllowedTools
    | OpenAIChatCompletionToolChoiceFunctionTool
    | OpenAIChatCompletionToolChoiceCustomTool,
    Field(discriminator="type"),
]

register_schema(OpenAIChatCompletionToolChoice, name="OpenAIChatCompletionToolChoice")


@json_schema_type
class OpenAITopLogProb(BaseModel):
    """The top log probability for a token from an OpenAI-compatible chat completion response."""

    token: str = Field(..., description="The token.")
    bytes: list[int] | None = Field(default=None, description="The bytes for the token.")
    logprob: float = Field(..., description="The log probability of the token.")


@json_schema_type
class OpenAITokenLogProb(BaseModel):
    """The log probability for a token from an OpenAI-compatible chat completion response."""

    token: str = Field(..., description="The token.")
    bytes: list[int] | None = Field(default=None, description="The bytes for the token.")
    logprob: float = Field(..., description="The log probability of the token.")
    top_logprobs: list[OpenAITopLogProb] | None = Field(
        default=None, description="The top log probabilities for the token."
    )


@json_schema_type
class OpenAIChoiceLogprobs(BaseModel):
    """The log probabilities for the tokens in the message from an OpenAI-compatible chat completion response."""

    content: list[OpenAITokenLogProb] | None = Field(
        default=None,
        description="The log probabilities for the tokens in the message.",
    )
    refusal: list[OpenAITokenLogProb] | None = Field(
        default=None,
        description="The log probabilities for the refusal tokens.",
    )


@json_schema_type
class OpenAIChoiceDelta(BaseModel):
    """A delta from an OpenAI-compatible chat completion streaming response."""

    content: str | None = Field(default=None, description="The content of the delta.")
    refusal: str | None = Field(default=None, description="The refusal of the delta.")
    role: Literal["developer", "system", "user", "assistant", "tool"] | None = Field(
        default=None,
        json_schema_extra=remove_null_from_anyof,
        description="The role of the delta.",
    )
    tool_calls: list[OpenAIChatCompletionToolCall] | None = Field(
        default=None,
        json_schema_extra=remove_null_from_anyof,
        description="The tool calls of the delta.",
    )
    reasoning_content: str | None = Field(
        default=None, description="The reasoning content from the model (for o1/o3 models)."
    )


# OpenAI finish_reason enum values
OpenAIFinishReason = Literal["stop", "length", "tool_calls", "content_filter", "function_call"]
register_schema(OpenAIFinishReason, name="OpenAIFinishReason")


@json_schema_type
class OpenAIChunkChoice(BaseModel):
    """A chunk choice from an OpenAI-compatible chat completion streaming response."""

    delta: OpenAIChoiceDelta = Field(..., description="The delta from the chunk.")
    finish_reason: OpenAIFinishReason | None = Field(
        default=None, json_schema_extra=nullable_openai_style, description="The reason the model stopped generating."
    )
    index: int = Field(..., ge=0, description="The index of the choice.")
    logprobs: OpenAIChoiceLogprobs | None = Field(
        default=None,
        json_schema_extra=remove_null_from_anyof,
        description="The log probabilities for the tokens in the message.",
    )


@json_schema_type
class OpenAIChatCompletionResponseMessage(BaseModel):
    """An assistant message returned in a chat completion response."""

    role: Literal["assistant"] = Field(
        default="assistant", description="The role of the message author, always 'assistant' in responses."
    )
    content: str | None = Field(default=None, description="The content of the message.")
    tool_calls: list[OpenAIChatCompletionToolCall | OpenAIChatCompletionCustomToolCall] | None = Field(
        default=None, json_schema_extra=remove_null_from_anyof, description="The tool calls generated by the model."
    )
    refusal: str | None = Field(default=None, description="The refusal message generated by the model.")
    function_call: OpenAIChatCompletionToolCallFunction | None = Field(
        default=None,
        json_schema_extra=remove_null_from_anyof,
        description="Deprecated: the name and arguments of a function that should be called.",
    )
    annotations: list[dict[str, Any]] | None = Field(
        default=None,
        json_schema_extra=remove_null_from_anyof,
        description="Annotations for the message, when applicable.",
    )
    audio: dict[str, Any] | None = Field(
        default=None, description="Audio response data when using audio output modality."
    )


@json_schema_type
class OpenAIChoice(BaseModel):
    """A choice from an OpenAI-compatible chat completion response."""

    message: OpenAIChatCompletionResponseMessage = Field(..., description="The message from the model.")
    finish_reason: OpenAIFinishReason = Field(..., description="The reason the model stopped generating.")
    index: int = Field(..., ge=0, description="The index of the choice.")
    logprobs: OpenAIChoiceLogprobs | None = Field(
        default=None,
        description="The log probabilities for the tokens in the message.",
    )


class OpenAIChatCompletionUsageCompletionTokensDetails(BaseModel):
    """Token details for output tokens in OpenAI chat completion usage."""

    reasoning_tokens: int = Field(default=0, ge=0, description="Number of tokens used for reasoning (o1/o3 models).")


class OpenAIChatCompletionUsagePromptTokensDetails(BaseModel):
    """Token details for prompt tokens in OpenAI chat completion usage."""

    cached_tokens: int = Field(default=0, ge=0, description="Number of tokens retrieved from cache.")


@json_schema_type
class OpenAIChatCompletionUsage(BaseModel):
    """Usage information for OpenAI chat completion."""

    prompt_tokens: int = Field(default=0, ge=0, description="Number of tokens in the prompt.")
    completion_tokens: int = Field(default=0, ge=0, description="Number of tokens in the completion.")
    total_tokens: int = Field(default=0, ge=0, description="Total tokens used (prompt + completion).")
    prompt_tokens_details: OpenAIChatCompletionUsagePromptTokensDetails | None = Field(
        default=None, json_schema_extra=remove_null_from_anyof, description="Detailed breakdown of input token usage."
    )
    completion_tokens_details: OpenAIChatCompletionUsageCompletionTokensDetails | None = Field(
        default=None, json_schema_extra=remove_null_from_anyof, description="Detailed breakdown of output token usage."
    )


@json_schema_type
class OpenAIChatCompletion(BaseModel):
    """Response from an OpenAI-compatible chat completion request."""

    id: str = Field(..., description="The ID of the chat completion.")
    choices: list[OpenAIChoice] = Field(..., min_length=1, description="List of choices.")
    object: Literal["chat.completion"] = Field(default="chat.completion", description="The object type.")
    created: int = Field(..., ge=0, description="The Unix timestamp in seconds when the chat completion was created.")
    model: str = Field(..., description="The model that was used to generate the chat completion.")
    service_tier: str | None = Field(default=None, description="The service tier that was used for this response.")
    system_fingerprint: str | None = Field(
        default=None, json_schema_extra=remove_null_from_anyof, description="System fingerprint for this completion."
    )
    usage: OpenAIChatCompletionUsage | None = Field(
        default=None,
        json_schema_extra=remove_null_from_anyof,
        description="Token usage information for the completion.",
    )


@json_schema_type
class OpenAIChatCompletionChunk(BaseModel):
    """Chunk from a streaming response to an OpenAI-compatible chat completion request."""

    id: str = Field(..., description="The ID of the chat completion.")
    choices: list[OpenAIChunkChoice] = Field(..., description="List of choices.")
    object: Literal["chat.completion.chunk"] = Field(default="chat.completion.chunk", description="The object type.")
    created: int = Field(..., ge=0, description="The Unix timestamp in seconds when the chat completion was created.")
    model: str = Field(..., description="The model that was used to generate the chat completion.")
    service_tier: str | None = Field(default=None, description="The service tier that was used for this response.")
    system_fingerprint: str | None = Field(
        default=None,
        json_schema_extra=remove_null_from_anyof,
        description="System fingerprint for this completion chunk.",
    )
    usage: OpenAIChatCompletionUsage | None = Field(
        default=None,
        json_schema_extra=remove_null_from_anyof,
        description="Token usage information (typically included in final chunk with stream_options).",
    )


@json_schema_type
class OpenAICompletionLogprobs(BaseModel):
    """The log probabilities for the tokens from an OpenAI-compatible completion response."""

    text_offset: list[int] | None = Field(default=None, description="The offset of the token in the text.")
    token_logprobs: list[float] | None = Field(default=None, description="The log probabilities for the tokens.")
    tokens: list[str] | None = Field(default=None, description="The tokens.")
    top_logprobs: list[dict[str, float]] | None = Field(
        default=None, description="The top log probabilities for the tokens."
    )


@json_schema_type
class OpenAICompletionChoice(BaseModel):
    """A choice from an OpenAI-compatible completion response."""

    finish_reason: OpenAIFinishReason = Field(..., description="The reason the model stopped generating.")
    text: str = Field(..., description="The text of the choice.")
    index: int = Field(..., ge=0, description="The index of the choice.")
    logprobs: OpenAIChoiceLogprobs | None = Field(
        default=None, description="The log probabilities for the tokens in the choice."
    )


@json_schema_type
class OpenAICompletion(BaseModel):
    """Response from an OpenAI-compatible completion request."""

    id: str = Field(..., description="The ID of the completion.")
    choices: list[OpenAICompletionChoice] = Field(..., min_length=1, description="List of choices.")
    created: int = Field(..., ge=0, description="The Unix timestamp in seconds when the completion was created.")
    model: str = Field(..., description="The model that was used to generate the completion.")
    object: Literal["text_completion"] = Field(default="text_completion", description="The object type.")


@json_schema_type
class OpenAIEmbeddingData(BaseModel):
    """A single embedding data object from an OpenAI-compatible embeddings response."""

    object: Literal["embedding"] = Field(default="embedding", description="The object type.")
    # TODO: consider dropping str and using openai.types.embeddings.Embedding instead of OpenAIEmbeddingData
    embedding: list[float] | str = Field(
        ...,
        description="The embedding vector as a list of floats (when encoding_format='float') or as a base64-encoded string.",
    )
    index: int = Field(..., ge=0, description="The index of the embedding in the input list.")


@json_schema_type
class OpenAIEmbeddingUsage(BaseModel):
    """Usage information for an OpenAI-compatible embeddings response."""

    prompt_tokens: int = Field(..., description="The number of tokens in the input.")
    total_tokens: int = Field(..., description="The total number of tokens used.")


@json_schema_type
class OpenAIEmbeddingsResponse(BaseModel):
    """Response from an OpenAI-compatible embeddings request."""

    object: Literal["list"] = Field(default="list", description="The object type.")
    data: list[OpenAIEmbeddingData] = Field(..., min_length=1, description="List of embedding data objects.")
    model: str = Field(..., description="The model that was used to generate the embeddings.")
    usage: OpenAIEmbeddingUsage = Field(..., description="Usage information.")


class TextTruncation(Enum):
    """Config for how to truncate text for embedding when text is longer than the model's max sequence length."""

    none = "none"
    start = "start"
    end = "end"


class EmbeddingTaskType(Enum):
    """How is the embedding being used? This is only supported by asymmetric embedding models."""

    query = "query"
    document = "document"


class ServiceTier(StrEnum):
    """The service tier for the request."""

    auto = "auto"
    default = "default"
    flex = "flex"
    priority = "priority"


class OpenAICompletionWithInputMessages(OpenAIChatCompletion):
    """Chat completion response extended with the original input messages."""

    input_messages: list[OpenAIMessageParam] = Field(
        ..., description="The input messages used to generate this completion."
    )


@json_schema_type
class ListOpenAIChatCompletionResponse(BaseModel):
    """Response from listing OpenAI-compatible chat completions."""

    data: list[OpenAICompletionWithInputMessages] = Field(
        ..., description="List of chat completion objects with their input messages."
    )
    has_more: bool = Field(..., description="Whether there are more completions available beyond this list.")
    first_id: str = Field(..., description="ID of the first completion in this list.")
    last_id: str = Field(..., description="ID of the last completion in this list.")
    object: Literal["list"] = Field(default="list", description="Must be 'list' to identify this as a list response.")


# extra_body can be accessed via .model_extra
@json_schema_type
class OpenAICompletionRequestWithExtraBody(BaseModel, extra="allow"):
    """Request parameters for OpenAI-compatible completion endpoint."""

    # Standard OpenAI completion parameters
    model: str = Field(..., description="The identifier of the model to use.")
    prompt: str | list[str] | list[int] | list[list[int]] = Field(
        ..., description="The prompt to generate a completion for."
    )
    best_of: int | None = Field(default=None, ge=1, description="The number of completions to generate.")
    echo: bool | None = Field(default=None, description="Whether to echo the prompt.")
    frequency_penalty: float | None = Field(
        default=None, ge=-2.0, le=2.0, description="The penalty for repeated tokens."
    )
    logit_bias: dict[str, float] | None = Field(default=None, description="The logit bias to use.")
    logprobs: bool | None = Field(default=None, description="The log probabilities to use.")
    max_tokens: int | None = Field(default=None, ge=1, description="The maximum number of tokens to generate.")
    n: int | None = Field(default=None, ge=1, description="The number of completions to generate.")
    presence_penalty: float | None = Field(
        default=None, ge=-2.0, le=2.0, description="The penalty for repeated tokens."
    )
    seed: int | None = Field(default=None, description="The seed to use.")
    stop: str | list[str] | None = Field(default=None, description="The stop tokens to use.")
    stream: bool | None = Field(default=None, description="Whether to stream the response.")
    stream_options: dict[str, Any] | None = Field(default=None, description="The stream options to use.")
    temperature: float | None = Field(default=None, ge=0.0, le=2.0, description="The temperature to use.")
    top_p: float | None = Field(default=None, ge=0.0, le=1.0, description="The top p to use.")
    user: str | None = Field(default=None, description="The user to use.")
    suffix: str | None = Field(default=None, description="The suffix that should be appended to the completion.")


# extra_body can be accessed via .model_extra
@json_schema_type
class OpenAIChatCompletionRequestWithExtraBody(BaseModel, extra="allow"):
    """Request parameters for OpenAI-compatible chat completion endpoint."""

    # Standard OpenAI chat completion parameters
    model: str = Field(..., description="The identifier of the model to use.")
    messages: Annotated[
        list[OpenAIMessageParam], Field(..., min_length=1, description="List of messages in the conversation.")
    ]
    frequency_penalty: float | None = Field(
        default=None, ge=-2.0, le=2.0, description="The penalty for repeated tokens."
    )
    function_call: str | dict[str, Any] | None = Field(default=None, description="The function call to use.")
    functions: list[dict[str, Any]] | None = Field(default=None, description="List of functions to use.")
    logit_bias: dict[str, float] | None = Field(default=None, description="The logit bias to use.")
    logprobs: bool | None = Field(default=None, description="The log probabilities to use.")
    max_completion_tokens: int | None = Field(
        default=None, ge=1, description="The maximum number of tokens to generate."
    )
    max_tokens: int | None = Field(default=None, ge=1, description="The maximum number of tokens to generate.")
    n: int | None = Field(default=None, ge=1, description="The number of completions to generate.")
    parallel_tool_calls: bool | None = Field(default=None, description="Whether to parallelize tool calls.")
    presence_penalty: float | None = Field(
        default=None, ge=-2.0, le=2.0, description="The penalty for repeated tokens."
    )
    response_format: OpenAIResponseFormatParam | None = Field(default=None, description="The response format to use.")
    seed: int | None = Field(default=None, description="The seed to use.")
    stop: str | list[str] | None = Field(default=None, description="The stop tokens to use.")
    stream: bool | None = Field(default=None, description="Whether to stream the response.")
    stream_options: dict[str, Any] | None = Field(default=None, description="The stream options to use.")
    temperature: float | None = Field(default=None, ge=0.0, le=2.0, description="The temperature to use.")
    tool_choice: str | dict[str, Any] | None = Field(default=None, description="The tool choice to use.")
    tools: list[dict[str, Any]] | None = Field(default=None, description="The tools to use.")
    top_logprobs: int | None = Field(
        default=None, ge=0, le=20, description="The number of most likely tokens to return at each position."
    )
    top_p: float | None = Field(default=None, ge=0.0, le=1.0, description="The top p to use.")
    user: str | None = Field(default=None, description="The user to use.")
    safety_identifier: str | None = Field(
        default=None,
        max_length=64,
        description="A stable identifier used for safety monitoring and abuse detection.",
    )
    service_tier: ServiceTier | None = Field(default=None, description="The service tier to use for this request.")
    reasoning_effort: Literal["none", "minimal", "low", "medium", "high", "xhigh"] | None = Field(
        default=None, description="The effort level for reasoning models."
    )
    prompt_cache_key: str | None = Field(
        default=None, max_length=64, description="A key to use when reading from or writing to the prompt cache."
    )


@json_schema_type
class OpenAIEmbeddingsRequestWithExtraBody(BaseModel, extra="allow"):
    """Request parameters for OpenAI-compatible embeddings endpoint."""

    model: str = Field(..., description="The identifier of the model to use.")
    input: (
        Annotated[str, Field(title="string")]
        | Annotated[list[str], Field(title="Array of strings", min_length=1, max_length=2048)]
        | Annotated[list[int], Field(title="Array of tokens", min_length=1, max_length=2048)]
        | Annotated[
            list[Annotated[list[int], Field(min_length=1)]],
            Field(title="Array of token arrays", min_length=1, max_length=2048),
        ]
    ) = Field(..., description="Input text to embed, encoded as a string or array of tokens.")
    encoding_format: Literal["float", "base64"] = Field(
        default="float", description="The format to return the embeddings in."
    )
    dimensions: int | None = Field(
        default=None,
        ge=1,
        description="The number of dimensions for output embeddings.",
        json_schema_extra=remove_null_from_anyof,
    )
    user: str | None = Field(
        default=None,
        description="A unique identifier representing your end-user.",
        json_schema_extra=remove_null_from_anyof,
    )

    @field_validator("dimensions", "user", mode="before")
    @classmethod
    def _reject_explicit_null(cls, v: Any, info: Any) -> Any:
        """Reject explicit null values to match OpenAI API behavior."""
        if v is None:
            raise ValueError(f"{info.field_name} cannot be null")
        return v


# New Request Models for Inference Endpoints
@json_schema_type
class ListChatCompletionsRequest(BaseModel):
    """Request model for listing chat completions."""

    after: str | None = Field(default=None, description="The ID of the last chat completion to return.")
    limit: int | None = Field(default=20, ge=1, description="The maximum number of chat completions to return.")
    model: str | None = Field(default=None, description="The model to filter by.")
    order: Order | None = Field(
        default=Order.desc,
        description='The order to sort the chat completions by: "asc" or "desc". Defaults to "desc".',
    )


@json_schema_type
class GetChatCompletionRequest(BaseModel):
    """Request model for getting a chat completion."""

    completion_id: str = Field(..., description="ID of the chat completion.")


@json_schema_type
class RerankRequest(BaseModel):
    """Request model for reranking documents."""

    model: str = Field(..., description="The identifier of the reranking model to use.")
    query: str | OpenAIChatCompletionContentPartTextParam | OpenAIChatCompletionContentPartImageParam = Field(
        ...,
        description="The search query to rank items against. Can be a string, text content part, or image content part.",
    )
    items: list[str | OpenAIChatCompletionContentPartTextParam | OpenAIChatCompletionContentPartImageParam] = Field(
        ...,
        min_length=1,
        description="List of items to rerank. Each item can be a string, text content part, or image content part.",
    )
    max_num_results: int | None = Field(
        default=None, ge=1, description="Maximum number of results to return. Default: returns all."
    )


class OpenAIChatCompletionWithReasoning(BaseModel):
    """Internal wrapper: a CC response with extracted reasoning content.

    Returned by openai_chat_completions_with_reasoning for non-streaming.
    The Responses layer unwraps .completion and reads .reasoning_content.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    completion: Any = Field(..., description="The chat completion response.")
    reasoning_content: str | None = Field(None, description="Extracted reasoning content, if any.")


class OpenAIChatCompletionChunkWithReasoning(BaseModel):
    """Internal wrapper: a CC streaming chunk with extracted reasoning content.

    Yielded by openai_chat_completions_with_reasoning for streaming.
    The Responses layer unwraps .chunk and reads .reasoning_content.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    chunk: Any = Field(..., description="The chat completion chunk.")
    reasoning_content: str | None = Field(None, description="Extracted reasoning content, if any.")


__all__ = [
    # Sampling
    "GreedySamplingStrategy",
    "TopPSamplingStrategy",
    "TopKSamplingStrategy",
    "SamplingStrategy",
    "SamplingParams",
    "LogProbConfig",
    # Quantization
    "QuantizationType",
    "Fp8QuantizationConfig",
    "Bf16QuantizationConfig",
    "Int4QuantizationConfig",
    "QuantizationConfig",
    # Messages
    "UserMessage",
    "SystemMessage",
    "ToolResponseMessage",
    "ToolChoice",
    "TokenLogProbs",
    # Response
    "ChatCompletionResponseEventType",
    "ResponseFormatType",
    "JsonSchemaResponseFormat",
    "GrammarResponseFormat",
    "ResponseFormat",
    "CompletionRequest",
    "SystemMessageBehavior",
    "EmbeddingsResponse",
    "RerankData",
    "RerankResponse",
    # OpenAI Compatibility
    "OpenAIChatCompletionContentPartTextParam",
    "OpenAIChatCompletionWithReasoning",
    "OpenAIChatCompletionChunkWithReasoning",
    "OpenAIImageURL",
    "OpenAIChatCompletionContentPartImageParam",
    "OpenAIFileFile",
    "OpenAIFile",
    "OpenAIChatCompletionContentPartParam",
    "OpenAIChatCompletionMessageContent",
    "OpenAIChatCompletionTextOnlyMessageContent",
    "OpenAIUserMessageParam",
    "OpenAISystemMessageParam",
    "OpenAIChatCompletionToolCallFunction",
    "OpenAIChatCompletionToolCall",
    "OpenAIChatCompletionCustomToolCallFunction",
    "OpenAIChatCompletionCustomToolCall",
    "OpenAIAssistantMessageParam",
    "OpenAIToolMessageParam",
    "OpenAIDeveloperMessageParam",
    "OpenAIMessageParam",
    "OpenAIResponseFormatText",
    "OpenAIJSONSchema",
    "OpenAIResponseFormatJSONSchema",
    "OpenAIResponseFormatJSONObject",
    "OpenAIResponseFormatParam",
    "FunctionToolConfig",
    "OpenAIChatCompletionToolChoiceFunctionTool",
    "CustomToolConfig",
    "OpenAIChatCompletionToolChoiceCustomTool",
    "AllowedToolsConfig",
    "OpenAIChatCompletionToolChoiceAllowedTools",
    "OpenAIChatCompletionToolChoice",
    "OpenAITopLogProb",
    "OpenAITokenLogProb",
    "OpenAIChoiceLogprobs",
    "OpenAIChoiceDelta",
    "OpenAIChunkChoice",
    "OpenAIChatCompletionResponseMessage",
    "OpenAIChoice",
    "OpenAIChatCompletionUsageCompletionTokensDetails",
    "OpenAIChatCompletionUsagePromptTokensDetails",
    "OpenAIChatCompletionUsage",
    "OpenAIChatCompletion",
    "OpenAIChatCompletionChunk",
    "OpenAICompletionLogprobs",
    "OpenAICompletionChoice",
    "OpenAICompletion",
    "OpenAIFinishReason",
    "OpenAIEmbeddingData",
    "OpenAIEmbeddingUsage",
    "OpenAIEmbeddingsResponse",
    "TextTruncation",
    "EmbeddingTaskType",
    "ServiceTier",
    "OpenAICompletionWithInputMessages",
    "ListOpenAIChatCompletionResponse",
    "OpenAICompletionRequestWithExtraBody",
    "OpenAIChatCompletionRequestWithExtraBody",
    "OpenAIEmbeddingsRequestWithExtraBody",
    # Request Models
    "ListChatCompletionsRequest",
    "GetChatCompletionRequest",
    "RerankRequest",
]
