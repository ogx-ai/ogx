# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

from openai import APIStatusError
from openai.types.chat import ChatCompletionToolParam
from opentelemetry import trace

from ogx.log import get_logger
from ogx.providers.utils.inference.openai_compat import convert_tooldef_to_openai_tool
from ogx.providers.utils.inference.prompt_adapter import interleaved_content_as_str
from ogx.providers.utils.tools.mcp import list_mcp_tools
from ogx_api import (
    AllowedToolsFilter,
    ApprovalFilter,
    Connectors,
    GetConnectorRequest,
    Inference,
    MCPListToolsTool,
    ModelNotFoundError,
    OpenAIAssistantMessageParam,
    OpenAIChatCompletion,
    OpenAIChatCompletionChunk,
    OpenAIChatCompletionChunkWithReasoning,
    OpenAIChatCompletionCustomToolCall,
    OpenAIChatCompletionRequestWithExtraBody,
    OpenAIChatCompletionResponseMessage,
    OpenAIChatCompletionToolCall,
    OpenAIChatCompletionToolChoice,
    OpenAIChatCompletionToolChoiceAllowedTools,
    OpenAIChatCompletionToolChoiceCustomTool,
    OpenAIChatCompletionToolChoiceFunctionTool,
    OpenAIChatCompletionUsage,
    OpenAIChatCompletionWithReasoning,
    OpenAIChoice,
    OpenAIChoiceLogprobs,
    OpenAIFinishReason,
    OpenAIMessageParam,
    OpenAIResponseContentPartOutputText,
    OpenAIResponseContentPartReasoningText,
    OpenAIResponseContentPartRefusal,
    OpenAIResponseError,
    OpenAIResponseIncompleteDetails,
    OpenAIResponseInputTool,
    OpenAIResponseInputToolChoice,
    OpenAIResponseInputToolChoiceAllowedTools,
    OpenAIResponseInputToolChoiceCustomTool,
    OpenAIResponseInputToolChoiceFileSearch,
    OpenAIResponseInputToolChoiceFunctionTool,
    OpenAIResponseInputToolChoiceMCPTool,
    OpenAIResponseInputToolChoiceMode,
    OpenAIResponseInputToolChoiceWebSearch,
    OpenAIResponseInputToolMCP,
    OpenAIResponseMCPApprovalRequest,
    OpenAIResponseMessage,
    OpenAIResponseObject,
    OpenAIResponseObjectStream,
    OpenAIResponseObjectStreamResponseCompleted,
    OpenAIResponseObjectStreamResponseContentPartAdded,
    OpenAIResponseObjectStreamResponseContentPartDone,
    OpenAIResponseObjectStreamResponseCreated,
    OpenAIResponseObjectStreamResponseFailed,
    OpenAIResponseObjectStreamResponseFunctionCallArgumentsDelta,
    OpenAIResponseObjectStreamResponseFunctionCallArgumentsDone,
    OpenAIResponseObjectStreamResponseIncomplete,
    OpenAIResponseObjectStreamResponseInProgress,
    OpenAIResponseObjectStreamResponseMcpCallArgumentsDelta,
    OpenAIResponseObjectStreamResponseMcpCallArgumentsDone,
    OpenAIResponseObjectStreamResponseMcpListToolsCompleted,
    OpenAIResponseObjectStreamResponseMcpListToolsInProgress,
    OpenAIResponseObjectStreamResponseOutputItemAdded,
    OpenAIResponseObjectStreamResponseOutputItemDone,
    OpenAIResponseObjectStreamResponseOutputTextDelta,
    OpenAIResponseObjectStreamResponseReasoningTextDelta,
    OpenAIResponseObjectStreamResponseReasoningTextDone,
    OpenAIResponseObjectStreamResponseRefusalDelta,
    OpenAIResponseObjectStreamResponseRefusalDone,
    OpenAIResponseOutput,
    OpenAIResponseOutputMessageContentOutputText,
    OpenAIResponseOutputMessageFileSearchToolCall,
    OpenAIResponseOutputMessageFunctionToolCall,
    OpenAIResponseOutputMessageMCPCall,
    OpenAIResponseOutputMessageMCPListTools,
    OpenAIResponseOutputMessageReasoningContent,
    OpenAIResponseOutputMessageReasoningItem,
    OpenAIResponseOutputMessageReasoningSummary,
    OpenAIResponseOutputMessageWebSearchToolCall,
    OpenAIResponsePrompt,
    OpenAIResponseReasoning,
    OpenAIResponseText,
    OpenAIResponseUsage,
    OpenAIResponseUsageInputTokensDetails,
    OpenAIResponseUsageOutputTokensDetails,
    OpenAIToolMessageParam,
    ResponseItemInclude,
    ResponseStreamOptions,
    ResponseTruncation,
    Safety,
    ToolDef,
    WebSearchToolTypes,
)
from ogx_api.inference import ServiceTier

from .types import (
    AssistantMessageWithReasoning,
    ChatCompletionContext,
    ChatCompletionResult,
)
from .utils import (
    convert_chat_choice_to_response_message,
    convert_mcp_tool_choice,
    is_function_tool_call,
    run_guardrails,
    should_summarize_reasoning,
    summarize_reasoning,
)

logger = get_logger(name=__name__, category="agents::builtin")
tracer = trace.get_tracer(__name__)

# Built-in tool names that the server knows how to execute itself.
# Anything else is either a registered function tool (client-side) or a hallucinated name.
_SERVER_SIDE_BUILTIN_TOOL_NAMES = frozenset({"web_search", "knowledge_search", "file_search"})

# Maps OpenAI Chat Completions error codes to Responses API error codes
_RESPONSES_API_ERROR_CODES = {
    "invalid_base64": "invalid_base64_image",
}

_VALID_RESPONSE_ERROR_CODES = frozenset(
    {
        "server_error",
        "rate_limit_exceeded",
        "invalid_prompt",
        "vector_store_timeout",
        "invalid_image",
        "invalid_image_format",
        "invalid_base64_image",
        "invalid_image_url",
        "image_too_large",
        "image_too_small",
        "image_parse_error",
        "image_content_policy_violation",
        "invalid_image_mode",
        "image_file_too_large",
        "unsupported_image_media_type",
        "empty_image_file",
        "failed_to_download_image",
        "image_file_not_found",
    }
)


def extract_openai_error(exc: Exception) -> tuple[str, str]:
    """Extract error code and message from a provider SDK exception.

    The exception should have a `body` attribute containing error details in one of two formats:
        1. Nested: {"error": {"code": "...", "message": "...", ...}}
        2. Direct: {"code": "...", "message": "...", ...}

    Args:
        exc: An exception with a `body` attribute containing error details

    Returns:
        Tuple of (error_code, error_message). Falls back to ("server_error", str(exc))
        if the body doesn't contain a valid code. The message is always preserved.
    """
    body = getattr(exc, "body", None)
    fallback_message = str(exc)

    if not isinstance(body, dict):
        logger.warning("Unexpected body type, expected dict", body_type=type(body), exc=exc)
        return ("server_error", fallback_message)

    # Try nested format first: {"error": {"code": "...", ...}}
    error_obj = body.get("error")
    if isinstance(error_obj, dict):
        raw_code = error_obj.get("code")
        raw_message = error_obj.get("message")
    else:
        raw_code = body.get("code")
        raw_message = body.get("message")

    if raw_code and isinstance(raw_code, str):
        final_code: str = _RESPONSES_API_ERROR_CODES[raw_code] if raw_code in _RESPONSES_API_ERROR_CODES else raw_code
    else:
        final_code = "server_error"

    if final_code not in _VALID_RESPONSE_ERROR_CODES:
        logger.info("Unmapped provider error code, falling back to server_error", final_code=final_code)
        final_code = "server_error"

    message: str = raw_message if isinstance(raw_message, str) else fallback_message

    return final_code, message


def convert_tooldef_to_chat_tool(tool_def):
    """Convert a ToolDef to OpenAI ChatCompletionToolParam format.

    Args:
        tool_def: ToolDef from the tools API

    Returns:
        ChatCompletionToolParam suitable for OpenAI chat completion
    """
    return convert_tooldef_to_openai_tool(
        tool_name=tool_def.name,
        description=tool_def.description,
        input_schema=tool_def.input_schema,
    )


class StreamingResponseOrchestrator:
    """Orchestrates streaming response generation with iterative tool calling and safety checks."""

    def __init__(
        self,
        inference_api: Inference,
        ctx: ChatCompletionContext,
        response_id: str,
        created_at: int,
        text: OpenAIResponseText,
        max_infer_iters: int,
        tool_executor,  # Will be the tool execution logic from the main class
        instructions: str | None,
        safety_api: Safety | None,
        guardrail_ids: list[str] | None = None,
        connectors_api: Connectors | None = None,
        prompt: OpenAIResponsePrompt | None = None,
        prompt_cache_key: str | None = None,
        parallel_tool_calls: bool | None = None,
        max_tool_calls: int | None = None,
        reasoning: OpenAIResponseReasoning | None = None,
        max_output_tokens: int | None = None,
        safety_identifier: str | None = None,
        service_tier: ServiceTier | None = None,
        metadata: dict[str, str] | None = None,
        include: list[ResponseItemInclude] | None = None,
        store: bool | None = True,
        truncation: ResponseTruncation | None = None,
        top_logprobs: int | None = None,
        presence_penalty: float | None = None,
        extra_body: dict | None = None,
        stream_options: ResponseStreamOptions | None = None,
    ):
        self.inference_api = inference_api
        self.ctx = ctx
        self.response_id = response_id
        self.created_at = created_at
        self.text = text
        self.max_infer_iters = max_infer_iters
        self.tool_executor = tool_executor
        self.safety_api = safety_api
        self.connectors_api = connectors_api
        self.guardrail_ids = guardrail_ids or []
        self.prompt = prompt
        self.prompt_cache_key = prompt_cache_key
        # System message that is inserted into the model's context
        self.instructions = instructions
        # Whether to allow more than one function tool call generated per turn.
        self.parallel_tool_calls = parallel_tool_calls
        # Max number of total calls to built-in tools that can be processed in a response
        self.max_tool_calls = max_tool_calls
        self.reasoning = reasoning
        # An upper bound for the number of tokens that can be generated for a response
        self.max_output_tokens = max_output_tokens
        self.safety_identifier = safety_identifier
        # Convert ServiceTier enum to string for internal storage
        # This allows us to update it with the actual tier returned by the provider
        self.service_tier = service_tier.value if service_tier is not None else None
        self.metadata = metadata
        self.truncation = truncation
        self.top_logprobs = top_logprobs
        self.stream_options = stream_options
        self.include = include
        self.extra_body = extra_body
        self.store = bool(store) if store is not None else True
        self.presence_penalty = presence_penalty
        self.sequence_number = 0
        # Store MCP tool mapping that gets built during tool processing
        self.mcp_tool_to_server: dict[str, OpenAIResponseInputToolMCP] = (
            ctx.tool_context.previous_tools if ctx.tool_context else {}
        )
        # Reverse mapping: server_label -> list of tool names for efficient lookup
        self.server_label_to_tools: dict[str, list[str]] = {}
        # Build initial reverse mapping from previous_tools
        for tool_name, mcp_server in self.mcp_tool_to_server.items():
            if mcp_server.server_label not in self.server_label_to_tools:
                self.server_label_to_tools[mcp_server.server_label] = []
            self.server_label_to_tools[mcp_server.server_label].append(tool_name)
        # Track final messages after all tool executions
        self.final_messages: list[OpenAIMessageParam] = []
        # mapping for annotations
        self.citation_files: dict[str, str] = {}
        # Track accumulated usage across all inference calls
        self.accumulated_usage: OpenAIResponseUsage | None = None
        # Track if we've sent a refusal response
        self.violation_detected = False
        # Track total calls made to built-in tools
        self.accumulated_builtin_tool_calls = 0
        # Track total output tokens generated across inference calls
        self.accumulated_builtin_output_tokens = 0

    async def _create_refusal_response(self, violation_message: str) -> OpenAIResponseObjectStream:
        """Create a refusal response to replace streaming content."""
        refusal_content = OpenAIResponseContentPartRefusal(refusal=violation_message)

        # Create a completed refusal response
        refusal_response = OpenAIResponseObject(
            background=False,
            id=self.response_id,
            created_at=self.created_at,
            frequency_penalty=self.ctx.frequency_penalty if self.ctx.frequency_penalty is not None else 0.0,
            model=self.ctx.model,
            status="completed",
            output=[OpenAIResponseMessage(role="assistant", content=[refusal_content], type="message")],
            temperature=self.ctx.temperature if self.ctx.temperature is not None else 1.0,
            top_p=self.ctx.top_p if self.ctx.top_p is not None else 1.0,
            top_logprobs=self.top_logprobs if self.top_logprobs is not None else 0,
            tools=self.ctx.available_tools(),
            tool_choice=self.ctx.tool_choice or OpenAIResponseInputToolChoiceMode.auto,
            truncation=self.truncation or "disabled",
            max_output_tokens=self.max_output_tokens,
            safety_identifier=self.safety_identifier,
            service_tier=self.service_tier or "default",
            metadata=self.metadata,
            presence_penalty=self.presence_penalty if self.presence_penalty is not None else 0.0,
            store=self.store,
            prompt_cache_key=self.prompt_cache_key,
        )

        self.sequence_number += 1
        return OpenAIResponseObjectStreamResponseCompleted(
            response=refusal_response, sequence_number=self.sequence_number
        )

    def _clone_outputs(self, outputs: list[OpenAIResponseOutput]) -> list[OpenAIResponseOutput]:
        cloned: list[OpenAIResponseOutput] = []
        for item in outputs:
            if hasattr(item, "model_copy"):
                cloned.append(item.model_copy(deep=True))
            else:
                cloned.append(item)
        return cloned

    def _snapshot_response(
        self,
        status: str,
        outputs: list[OpenAIResponseOutput],
        *,
        error: OpenAIResponseError | None = None,
        incomplete_details: OpenAIResponseIncompleteDetails | None = None,
    ) -> OpenAIResponseObject:
        completed_at = int(time.time()) if status == "completed" else None
        return OpenAIResponseObject(
            background=False,
            created_at=self.created_at,
            completed_at=completed_at,
            frequency_penalty=self.ctx.frequency_penalty if self.ctx.frequency_penalty is not None else 0.0,
            id=self.response_id,
            model=self.ctx.model,
            object="response",
            status=status,
            output=self._clone_outputs(outputs),
            text=self.text,
            temperature=self.ctx.temperature if self.ctx.temperature is not None else 1.0,
            top_p=self.ctx.top_p if self.ctx.top_p is not None else 1.0,
            top_logprobs=self.top_logprobs if self.top_logprobs is not None else 0,
            tools=self.ctx.available_tools(),
            tool_choice=self.ctx.tool_choice or OpenAIResponseInputToolChoiceMode.auto,
            error=error,
            incomplete_details=incomplete_details,
            usage=self.accumulated_usage,
            instructions=self.instructions,
            prompt=self.prompt,
            parallel_tool_calls=self.parallel_tool_calls,
            max_tool_calls=self.max_tool_calls,
            reasoning=self.reasoning,
            max_output_tokens=self.max_output_tokens,
            safety_identifier=self.safety_identifier,
            service_tier=self.service_tier or "default",
            metadata=self.metadata,
            truncation=self.truncation or "disabled",
            presence_penalty=self.presence_penalty if self.presence_penalty is not None else 0.0,
            store=self.store,
            prompt_cache_key=self.prompt_cache_key,
        )

    async def create_response(self) -> AsyncIterator[OpenAIResponseObjectStream]:
        output_messages: list[OpenAIResponseOutput] = []

        # Emit response.created followed by response.in_progress to align with OpenAI streaming
        yield OpenAIResponseObjectStreamResponseCreated(
            response=self._snapshot_response("in_progress", output_messages),
            sequence_number=self.sequence_number,
        )

        self.sequence_number += 1
        yield OpenAIResponseObjectStreamResponseInProgress(
            response=self._snapshot_response("in_progress", output_messages),
            sequence_number=self.sequence_number,
        )

        # Input safety validation - check messages before processing
        if self.guardrail_ids:
            combined_text = interleaved_content_as_str([msg.content for msg in self.ctx.messages])
            input_violation_message = await run_guardrails(self.safety_api, combined_text, self.guardrail_ids)
            if input_violation_message:
                logger.info("Input guardrail violation", input_violation_message=input_violation_message)
                yield await self._create_refusal_response(input_violation_message)
                return

        # Only 'disabled' truncation is supported for now
        # TODO: Implement actual truncation logic when 'auto' mode is supported
        if self.truncation == ResponseTruncation.auto:
            logger.warning("Truncation mode 'auto' is not yet supported")
            self.sequence_number += 1
            error = OpenAIResponseError(
                code="server_error",
                message="Truncation mode 'auto' is not supported. Use 'disabled' to let the inference provider reject oversized contexts.",
            )
            failure_response = self._snapshot_response("failed", output_messages, error=error)
            yield OpenAIResponseObjectStreamResponseFailed(
                response=failure_response,
                sequence_number=self.sequence_number,
            )
            return

        async for stream_event in self._process_tools(output_messages):
            yield stream_event

        chat_tool_choice: str | dict[str, Any] | None = None
        # Track allowed tools for filtering (persists across iterations)
        allowed_tool_names: set[str] | None = None
        # check truthiness of self.ctx.chat_tools to avoid len(None)
        if self.ctx.tool_choice and self.ctx.chat_tools:
            processed_tool_choice = await _process_tool_choice(
                self.ctx.chat_tools,
                self.ctx.tool_choice,
                self.server_label_to_tools,
            )
            # chat_tool_choice can be str, dict-like object, or None
            if isinstance(processed_tool_choice, str | type(None)):
                chat_tool_choice = processed_tool_choice
            elif isinstance(processed_tool_choice, OpenAIChatCompletionToolChoiceAllowedTools):
                # For allowed_tools: filter the tools list instead of using tool_choice
                # This maintains the constraint across all iterations while letting model
                # decide freely whether to call a tool or respond
                allowed_tool_names = {
                    tool["function"]["name"]
                    for tool in processed_tool_choice.allowed_tools.tools
                    if tool.get("type") == "function" and "function" in tool
                }
                # Use the mode (e.g., "required") for first iteration, then "auto"
                chat_tool_choice = (
                    processed_tool_choice.allowed_tools.mode if processed_tool_choice.allowed_tools.mode else "auto"
                )
            else:
                chat_tool_choice = processed_tool_choice.model_dump()

        n_iter = 0
        messages = self.ctx.messages.copy()
        final_status = "completed"
        incomplete_reason: str | None = None
        last_completion_result: ChatCompletionResult | None = None

        try:
            while True:
                if (
                    self.max_output_tokens is not None
                    and self.accumulated_builtin_output_tokens >= self.max_output_tokens
                ):
                    logger.info(
                        "Skipping inference call since max_output_tokens reached: /",
                        accumulated_builtin_output_tokens=self.accumulated_builtin_output_tokens,
                        max_output_tokens=self.max_output_tokens,
                    )
                    final_status = "incomplete"
                    incomplete_reason = "max_output_tokens"
                    break

                remaining_output_tokens = (
                    self.max_output_tokens - self.accumulated_builtin_output_tokens
                    if self.max_output_tokens is not None
                    else None
                )
                # Text is the default response format for chat completion so don't need to pass it
                # (some providers don't support non-empty response_format when tools are present)
                response_format = (
                    None if getattr(self.ctx.response_format, "type", None) == "text" else self.ctx.response_format
                )
                # Filter tools to only allowed ones if tool_choice specified an allowed list
                effective_tools = self.ctx.chat_tools
                if allowed_tool_names is not None and self.ctx.chat_tools is not None:
                    effective_tools = [
                        tool
                        for tool in self.ctx.chat_tools
                        if tool.get("function", {}).get("name") in allowed_tool_names
                    ]
                logger.debug("calling openai_chat_completion with tools", effective_tools=effective_tools)

                logprobs = (
                    True
                    if (self.include and ResponseItemInclude.message_output_text_logprobs in self.include)
                    or self.top_logprobs
                    else None
                )

                # In OpenAI, parallel_tool_calls is only allowed when 'tools' are specified.
                effective_parallel_tool_calls = (
                    self.parallel_tool_calls if effective_tools is not None and len(effective_tools) > 0 else None
                )

                # Merge user stream_options with default include_usage
                effective_stream_options = {"include_usage": True}
                if self.stream_options:
                    effective_stream_options.update(self.stream_options)

                params = OpenAIChatCompletionRequestWithExtraBody(
                    model=self.ctx.model,
                    messages=messages,
                    # Pydantic models are dict-compatible but mypy treats them as distinct types
                    tools=effective_tools,  # type: ignore[arg-type]
                    tool_choice=chat_tool_choice,
                    stream=True,
                    temperature=self.ctx.temperature,
                    top_p=self.ctx.top_p,
                    frequency_penalty=self.ctx.frequency_penalty,
                    response_format=response_format,
                    stream_options=effective_stream_options,
                    logprobs=logprobs,
                    parallel_tool_calls=effective_parallel_tool_calls,
                    reasoning_effort=self.reasoning.effort if self.reasoning else None,
                    safety_identifier=self.safety_identifier,
                    service_tier=ServiceTier(self.service_tier) if self.service_tier else None,
                    max_completion_tokens=remaining_output_tokens,
                    prompt_cache_key=self.prompt_cache_key,
                    top_logprobs=self.top_logprobs,
                    presence_penalty=self.presence_penalty,
                    **(self.extra_body or {}),
                )
                # Use reasoning-aware method when reasoning is explicitly requested
                completion_result: (
                    OpenAIChatCompletion
                    | AsyncIterator[OpenAIChatCompletionChunk]
                    | OpenAIChatCompletionWithReasoning
                    | AsyncIterator[OpenAIChatCompletionChunkWithReasoning]
                )
                if self.reasoning and self.reasoning.effort and self.reasoning.effort != "none":
                    try:
                        # Pass a copy — the router mutates params.model (strips provider prefix).
                        # Keep original params intact in-case of fallback to regular CC.
                        # NOTE : Is a deep-copy necessary ?
                        completion_result = await self.inference_api.openai_chat_completions_with_reasoning(
                            params.model_copy()
                        )
                    except (NotImplementedError, AttributeError, ValueError):
                        logger.critical(
                            "Provider does not support reasoning in chat completions. "
                            "Falling back to regular chat completion."
                        )
                        completion_result = await self.inference_api.openai_chat_completion(params)
                else:
                    completion_result = await self.inference_api.openai_chat_completion(params)

                # Process streaming chunks and build complete response
                completion_result_data = None
                async for stream_event_or_result in self._process_streaming_chunks(completion_result, output_messages):
                    if isinstance(stream_event_or_result, ChatCompletionResult):
                        completion_result_data = stream_event_or_result
                    else:
                        yield stream_event_or_result
                # If violation detected, skip the rest of processing since we already sent refusal
                if self.violation_detected:
                    return

                if not completion_result_data:
                    raise ValueError("Streaming chunk processor failed to return completion data")
                last_completion_result = completion_result_data

                if completion_result_data.service_tier is None:
                    # Since the default service_tier is "auto" and the returned service_tier is the
                    # "mode actually used, here we are setting output value for service_tier as "default"
                    # when service_tier is not supported.
                    logger.warning("Service tier is None, setting to default")
                    self.service_tier = "default"
                else:
                    # Update service_tier with actual value from provider response
                    # This is especially important when "auto" was used as input
                    self.service_tier = completion_result_data.service_tier
                current_response = self._build_chat_completion(completion_result_data)
                (
                    function_tool_calls,
                    non_function_tool_calls,
                    approvals,
                    next_turn_messages,
                ) = self._separate_tool_calls(current_response, messages, completion_result_data.reasoning_content)
                # add any approval requests required
                for tool_call in approvals:
                    async for evt in self._add_mcp_approval_request(
                        tool_call.function.name, tool_call.function.arguments, output_messages
                    ):
                        yield evt

                # Reasoning is independent of the response type — whether the assistant
                # response is a tool call or a content message, reasoning (if present)
                # is added to output_messages before processing choices.
                # The reasoning_content field is populated by the provider via
                # openai_chat_completions_with_reasoning, which maps provider-specific
                # reasoning fields to the standard reasoning_content attribute.
                if completion_result_data.reasoning_content:
                    reasoning_item = OpenAIResponseOutputMessageReasoningItem(
                        id=f"rs_{uuid.uuid4().hex}",
                        summary=[],
                        content=[
                            OpenAIResponseOutputMessageReasoningContent(text=completion_result_data.reasoning_content)
                        ],
                        status="in_progress",
                    )
                    reasoning_output_index = len(output_messages)
                    output_messages.append(reasoning_item)

                    self.sequence_number += 1
                    yield OpenAIResponseObjectStreamResponseOutputItemAdded(
                        response_id=self.response_id,
                        item=reasoning_item,
                        output_index=reasoning_output_index,
                        sequence_number=self.sequence_number,
                    )

                    if should_summarize_reasoning(self.reasoning):
                        summary_mode = (
                            self.reasoning.summary if self.reasoning and self.reasoning.summary else "concise"
                        )
                        summary_usage_list: list[OpenAIChatCompletionUsage] = []
                        summary_text = await summarize_reasoning(
                            inference_api=self.inference_api,
                            model=self.ctx.model,
                            reasoning_text=completion_result_data.reasoning_content,
                            summary_mode=summary_mode,
                            summary_usage=summary_usage_list,
                        )
                        if summary_text:
                            reasoning_item.summary = [OpenAIResponseOutputMessageReasoningSummary(text=summary_text)]
                        for usage in summary_usage_list:
                            self._accumulate_usage(usage)

                    reasoning_item.status = "completed"
                    self.sequence_number += 1
                    yield OpenAIResponseObjectStreamResponseOutputItemDone(
                        response_id=self.response_id,
                        item=reasoning_item,
                        output_index=reasoning_output_index,
                        sequence_number=self.sequence_number,
                    )

                for choice in current_response.choices:
                    has_tool_calls = choice.message.tool_calls and self.ctx.response_tools
                    if not has_tool_calls:
                        output_messages.append(
                            await convert_chat_choice_to_response_message(
                                choice,
                                self.citation_files,
                                message_id=completion_result_data.message_item_id,
                            )
                        )
                # Execute tool calls and coordinate results
                async for stream_event in self._coordinate_tool_execution(
                    function_tool_calls,
                    non_function_tool_calls,
                    completion_result_data,
                    output_messages,
                    next_turn_messages,
                ):
                    yield stream_event
                messages = next_turn_messages
                if not function_tool_calls and not non_function_tool_calls:
                    break

                if function_tool_calls:
                    logger.info("Exiting inference loop since there is a function (client-side) tool call")
                    break

                n_iter += 1
                # After first iteration, reset tool_choice to "auto" to let model decide freely
                # based on tool results (prevents infinite loops when forcing specific tools)
                # Note: When allowed_tool_names is set, tools are already filtered so model
                # can only call allowed tools - we just need to let it decide whether to call
                # a tool or respond (hence "auto" mode)
                if n_iter == 1 and chat_tool_choice and chat_tool_choice != "auto":
                    chat_tool_choice = "auto"

                if n_iter >= self.max_infer_iters:
                    logger.info(
                        "Exiting inference loop since iteration count() exceeds",
                        n_iter=n_iter,
                        max_infer_iters=self.max_infer_iters,
                    )
                    final_status = "incomplete"
                    incomplete_reason = "max_iterations_exceeded"
                    break

            if last_completion_result and last_completion_result.finish_reason == "length":
                final_status = "incomplete"
                incomplete_reason = "length"

        except ModelNotFoundError:
            raise
        except Exception as exc:  # noqa: BLE001
            self.final_messages = messages.copy()
            self.sequence_number += 1

            if isinstance(exc, APIStatusError) or (hasattr(exc, "status_code") and hasattr(exc, "body")):
                logger.warning("Provider SDK error during response generation", exc=exc)
                error_code, error_message = extract_openai_error(exc)
            else:
                logger.exception(
                    "Unexpected error during response generation", error_type=type(exc).__name__, exc_info=exc
                )
                error_code, error_message = (
                    "server_error",
                    "An unexpected error occurred while generating the response.",
                )
            error = OpenAIResponseError(code=error_code, message=error_message)
            failure_response = self._snapshot_response("failed", output_messages, error=error)
            yield OpenAIResponseObjectStreamResponseFailed(
                response=failure_response,
                sequence_number=self.sequence_number,
            )
            return

        self.final_messages = messages.copy()

        if final_status == "incomplete":
            self.sequence_number += 1
            incomplete_details = (
                OpenAIResponseIncompleteDetails(reason=incomplete_reason) if incomplete_reason else None
            )
            final_response = self._snapshot_response(
                "incomplete", output_messages, incomplete_details=incomplete_details
            )
            yield OpenAIResponseObjectStreamResponseIncomplete(
                response=final_response,
                sequence_number=self.sequence_number,
            )
        else:
            self.sequence_number += 1
            final_response = self._snapshot_response("completed", output_messages)
            yield OpenAIResponseObjectStreamResponseCompleted(
                response=final_response, sequence_number=self.sequence_number
            )

    def _separate_tool_calls(
        self, current_response, messages, reasoning_content: str | None = None
    ) -> tuple[list, list, list, list]:
        """Separate tool calls into function and non-function categories."""
        function_tool_calls = []
        non_function_tool_calls = []
        approvals = []
        next_turn_messages = messages.copy()

        for choice in current_response.choices:
            # Convert response message to input message format for multi-turn.
            # Use AssistantMessageWithReasoning if reasoning was present in the
            # CC response. Providers will be check for this AssistantMessageWithReasoning
            # message
            if reasoning_content:
                message = AssistantMessageWithReasoning(
                    content=choice.message.content,
                    tool_calls=choice.message.tool_calls,
                    reasoning_content=reasoning_content,
                )
            else:
                message = OpenAIAssistantMessageParam(  # type: ignore[assignment]
                    content=choice.message.content,
                    tool_calls=choice.message.tool_calls,
                )
            next_turn_messages.append(message)
            logger.debug("Choice message content", content=choice.message.content)
            logger.debug("Choice message tool_calls", tool_calls=choice.message.tool_calls)

            if choice.message.tool_calls and self.ctx.response_tools:
                executed_tool_calls: list = []
                has_deferred_or_denied = False
                for tool_call in choice.message.tool_calls:
                    if is_function_tool_call(tool_call, self.ctx.response_tools):
                        function_tool_calls.append(tool_call)
                        executed_tool_calls.append(tool_call)
                    elif (
                        tool_call.function
                        and tool_call.function.name not in _SERVER_SIDE_BUILTIN_TOOL_NAMES
                        and tool_call.function.name not in self.mcp_tool_to_server
                    ):
                        # The model called a tool name that is neither a registered function tool,
                        # nor a server-side built-in, nor an MCP tool — it hallucinated a name.
                        # Return it to the client as a function_call output item rather than
                        # crashing the server with an unhandled ValueError.
                        logger.warning(
                            "Model called unrecognized tool ; treating as a client-side function call.",
                            name=tool_call.function.name,
                        )
                        function_tool_calls.append(tool_call)
                        executed_tool_calls.append(tool_call)
                    else:
                        if self._approval_required(tool_call.function.name):
                            approval_response = self.ctx.approval_response(
                                tool_call.function.name, tool_call.function.arguments
                            )
                            if approval_response:
                                if approval_response.approve:
                                    logger.info(
                                        "Approval granted for on", id=tool_call.id, name=tool_call.function.name
                                    )
                                    non_function_tool_calls.append(tool_call)
                                    executed_tool_calls.append(tool_call)
                                else:
                                    logger.info("Approval denied", id=tool_call.id, name=tool_call.function.name)
                                    has_deferred_or_denied = True
                            else:
                                logger.info("Requesting approval for on", id=tool_call.id, name=tool_call.function.name)
                                approvals.append(tool_call)
                                has_deferred_or_denied = True
                        else:
                            non_function_tool_calls.append(tool_call)
                            executed_tool_calls.append(tool_call)
                if has_deferred_or_denied:
                    if executed_tool_calls:
                        next_turn_messages[-1] = OpenAIAssistantMessageParam(
                            content=choice.message.content,
                            tool_calls=executed_tool_calls,
                        )
                    else:
                        next_turn_messages.pop()

        return function_tool_calls, non_function_tool_calls, approvals, next_turn_messages

    def _accumulate_chunk_usage(self, chunk: OpenAIChatCompletionChunk) -> None:
        """Accumulate usage from a streaming chunk into the response usage format."""
        if not chunk.usage:
            return
        self._accumulate_usage(chunk.usage)

    def _accumulate_usage(self, usage: OpenAIChatCompletionUsage) -> None:
        """Accumulate chat completion usage into the response usage format."""
        self.accumulated_builtin_output_tokens += usage.completion_tokens

        if self.accumulated_usage is None:
            # Convert from chat completion format to response format
            self.accumulated_usage = OpenAIResponseUsage(
                input_tokens=usage.prompt_tokens,
                output_tokens=usage.completion_tokens,
                total_tokens=usage.total_tokens,
                input_tokens_details=OpenAIResponseUsageInputTokensDetails(
                    cached_tokens=usage.prompt_tokens_details.cached_tokens
                    if usage.prompt_tokens_details and usage.prompt_tokens_details.cached_tokens is not None
                    else 0
                ),
                output_tokens_details=OpenAIResponseUsageOutputTokensDetails(
                    reasoning_tokens=usage.completion_tokens_details.reasoning_tokens
                    if usage.completion_tokens_details and usage.completion_tokens_details.reasoning_tokens is not None
                    else 0
                ),
            )
        else:
            # Accumulate across multiple inference calls
            self.accumulated_usage = OpenAIResponseUsage(
                input_tokens=self.accumulated_usage.input_tokens + usage.prompt_tokens,
                output_tokens=self.accumulated_usage.output_tokens + usage.completion_tokens,
                total_tokens=self.accumulated_usage.total_tokens + usage.total_tokens,
                input_tokens_details=OpenAIResponseUsageInputTokensDetails(
                    cached_tokens=usage.prompt_tokens_details.cached_tokens
                    if usage.prompt_tokens_details and usage.prompt_tokens_details.cached_tokens is not None
                    else self.accumulated_usage.input_tokens_details.cached_tokens
                ),
                output_tokens_details=OpenAIResponseUsageOutputTokensDetails(
                    reasoning_tokens=usage.completion_tokens_details.reasoning_tokens
                    if usage.completion_tokens_details and usage.completion_tokens_details.reasoning_tokens is not None
                    else self.accumulated_usage.output_tokens_details.reasoning_tokens
                ),
            )

    async def _handle_reasoning_content_chunk(
        self,
        reasoning_content: str,
        reasoning_part_emitted: bool,
        reasoning_content_index: int,
        message_item_id: str,
        message_output_index: int,
    ) -> AsyncIterator[OpenAIResponseObjectStream]:
        # Emit content_part.added event for first reasoning chunk
        if not reasoning_part_emitted:
            self.sequence_number += 1
            yield OpenAIResponseObjectStreamResponseContentPartAdded(
                content_index=reasoning_content_index,
                response_id=self.response_id,
                item_id=message_item_id,
                output_index=message_output_index,
                part=OpenAIResponseContentPartReasoningText(
                    text="",  # Will be filled incrementally via reasoning deltas
                ),
                sequence_number=self.sequence_number,
            )
        # Emit reasoning_text.delta event
        self.sequence_number += 1
        yield OpenAIResponseObjectStreamResponseReasoningTextDelta(
            content_index=reasoning_content_index,
            delta=reasoning_content,
            item_id=message_item_id,
            output_index=message_output_index,
            sequence_number=self.sequence_number,
        )

    async def _handle_refusal_content_chunk(
        self,
        refusal_content: str,
        refusal_part_emitted: bool,
        refusal_content_index: int,
        message_item_id: str,
        message_output_index: int,
    ) -> AsyncIterator[OpenAIResponseObjectStream]:
        # Emit content_part.added event for first refusal chunk
        if not refusal_part_emitted:
            self.sequence_number += 1
            yield OpenAIResponseObjectStreamResponseContentPartAdded(
                content_index=refusal_content_index,
                response_id=self.response_id,
                item_id=message_item_id,
                output_index=message_output_index,
                part=OpenAIResponseContentPartRefusal(
                    refusal="",  # Will be filled incrementally via refusal deltas
                ),
                sequence_number=self.sequence_number,
            )
        # Emit refusal.delta event
        self.sequence_number += 1
        yield OpenAIResponseObjectStreamResponseRefusalDelta(
            content_index=refusal_content_index,
            delta=refusal_content,
            item_id=message_item_id,
            output_index=message_output_index,
            sequence_number=self.sequence_number,
        )

    async def _emit_reasoning_done_events(
        self,
        reasoning_text_accumulated: list[str],
        reasoning_content_index: int,
        message_item_id: str,
        message_output_index: int,
    ) -> AsyncIterator[OpenAIResponseObjectStream]:
        final_reasoning_text = "".join(reasoning_text_accumulated)
        # Emit reasoning_text.done event
        self.sequence_number += 1
        yield OpenAIResponseObjectStreamResponseReasoningTextDone(
            content_index=reasoning_content_index,
            text=final_reasoning_text,
            item_id=message_item_id,
            output_index=message_output_index,
            sequence_number=self.sequence_number,
        )
        # Emit content_part.done for reasoning
        self.sequence_number += 1
        yield OpenAIResponseObjectStreamResponseContentPartDone(
            content_index=reasoning_content_index,
            response_id=self.response_id,
            item_id=message_item_id,
            output_index=message_output_index,
            part=OpenAIResponseContentPartReasoningText(
                text=final_reasoning_text,
            ),
            sequence_number=self.sequence_number,
        )

    async def _emit_refusal_done_events(
        self,
        refusal_text_accumulated: list[str],
        refusal_content_index: int,
        message_item_id: str,
        message_output_index: int,
    ) -> AsyncIterator[OpenAIResponseObjectStream]:
        final_refusal_text = "".join(refusal_text_accumulated)
        # Emit refusal.done event
        self.sequence_number += 1
        yield OpenAIResponseObjectStreamResponseRefusalDone(
            content_index=refusal_content_index,
            refusal=final_refusal_text,
            item_id=message_item_id,
            output_index=message_output_index,
            sequence_number=self.sequence_number,
        )
        # Emit content_part.done for refusal
        self.sequence_number += 1
        yield OpenAIResponseObjectStreamResponseContentPartDone(
            content_index=refusal_content_index,
            response_id=self.response_id,
            item_id=message_item_id,
            output_index=message_output_index,
            part=OpenAIResponseContentPartRefusal(
                refusal=final_refusal_text,
            ),
            sequence_number=self.sequence_number,
        )

    async def _process_streaming_chunks(
        self, completion_result, output_messages: list[OpenAIResponseOutput]
    ) -> AsyncIterator[OpenAIResponseObjectStream | ChatCompletionResult]:
        """Process streaming chunks and emit events, returning completion data."""
        # Initialize result tracking
        chat_response_id = ""
        chat_response_content = []
        chat_response_tool_calls: dict[int, OpenAIChatCompletionToolCall] = {}
        chunk_created = 0
        chunk_model = ""
        chunk_finish_reason: OpenAIFinishReason = "stop"
        chat_response_logprobs = []
        chunk_service_tier: str | None = None

        # Create a placeholder message item for delta events
        message_item_id = f"msg_{uuid.uuid4()}"
        # Track tool call items for streaming events
        tool_call_item_ids: dict[int, str] = {}
        # Track content parts for streaming events
        message_item_added_emitted = False
        content_part_emitted = False
        reasoning_part_emitted = False
        refusal_part_emitted = False
        content_index = 0
        reasoning_content_index = 1  # reasoning is a separate content part
        refusal_content_index = 2  # refusal is a separate content part
        message_output_index = len(output_messages)
        reasoning_text_accumulated = []
        refusal_text_accumulated = []

        async for raw_chunk in completion_result:
            # Providers returning OpenAIChatCompletionChunkWithReasoning wrap
            # the chunk with a typed reasoning_content field. Unwrap here.
            if isinstance(raw_chunk, OpenAIChatCompletionChunkWithReasoning):
                chunk = raw_chunk.chunk
                reasoning_content = raw_chunk.reasoning_content
            else:
                chunk = raw_chunk
                reasoning_content = None

            chat_response_id = chunk.id
            chunk_created = chunk.created
            chunk_model = chunk.model
            # Extract service_tier from chunk if available (may be in final chunk)
            if chunk.service_tier is not None:
                chunk_service_tier = chunk.service_tier

            # Accumulate usage from chunks (typically in final chunk with stream_options)
            self._accumulate_chunk_usage(chunk)

            # Track deltas for this specific chunk for guardrail validation
            chunk_events: list[OpenAIResponseObjectStream] = []

            for chunk_choice in chunk.choices:
                # Collect logprobs if present
                chunk_logprobs = None
                if chunk_choice.logprobs and chunk_choice.logprobs.content:
                    chunk_logprobs = chunk_choice.logprobs.content
                    chat_response_logprobs.extend(chunk_logprobs)

                # Emit incremental text content as delta events
                if chunk_choice.delta.content:
                    # Emit output_item.added for the message on first content
                    if not message_item_added_emitted:
                        message_item_added_emitted = True
                        self.sequence_number += 1
                        message_item = OpenAIResponseMessage(
                            id=message_item_id,
                            content=[],
                            role="assistant",
                            status="in_progress",
                        )
                        yield OpenAIResponseObjectStreamResponseOutputItemAdded(
                            response_id=self.response_id,
                            item=message_item,
                            output_index=message_output_index,
                            sequence_number=self.sequence_number,
                        )

                    # Emit content_part.added event for first text chunk
                    if not content_part_emitted:
                        content_part_emitted = True
                        self.sequence_number += 1
                        yield OpenAIResponseObjectStreamResponseContentPartAdded(
                            content_index=content_index,
                            response_id=self.response_id,
                            item_id=message_item_id,
                            output_index=message_output_index,
                            part=OpenAIResponseContentPartOutputText(
                                text="",  # Will be filled incrementally via text deltas
                                logprobs=[],
                            ),
                            sequence_number=self.sequence_number,
                        )
                    self.sequence_number += 1

                    text_delta_event = OpenAIResponseObjectStreamResponseOutputTextDelta(
                        content_index=content_index,
                        delta=chunk_choice.delta.content,
                        item_id=message_item_id,
                        logprobs=chunk_logprobs if chunk_logprobs is not None else [],
                        output_index=message_output_index,
                        sequence_number=self.sequence_number,
                    )
                    # Buffer text delta events for guardrail check
                    if self.guardrail_ids:
                        chunk_events.append(text_delta_event)
                    else:
                        yield text_delta_event

                # Collect content for final response
                chat_response_content.append(chunk_choice.delta.content or "")
                if chunk_choice.finish_reason:
                    chunk_finish_reason = chunk_choice.finish_reason

                # Handle reasoning content if present.
                # reasoning_content comes from the typed wrapper
                # (OpenAIChatCompletionChunkWithReasoning) unwrapped above.
                if reasoning_content:
                    async for event in self._handle_reasoning_content_chunk(
                        reasoning_content=reasoning_content,
                        reasoning_part_emitted=reasoning_part_emitted,
                        reasoning_content_index=reasoning_content_index,
                        message_item_id=message_item_id,
                        message_output_index=message_output_index,
                    ):
                        # Buffer reasoning events for guardrail check
                        if self.guardrail_ids:
                            chunk_events.append(event)
                        else:
                            yield event
                    reasoning_part_emitted = True
                    reasoning_text_accumulated.append(reasoning_content)

                # Handle refusal content if present
                if chunk_choice.delta.refusal:
                    async for event in self._handle_refusal_content_chunk(
                        refusal_content=chunk_choice.delta.refusal,
                        refusal_part_emitted=refusal_part_emitted,
                        refusal_content_index=refusal_content_index,
                        message_item_id=message_item_id,
                        message_output_index=message_output_index,
                    ):
                        yield event
                    refusal_part_emitted = True
                    refusal_text_accumulated.append(chunk_choice.delta.refusal)

                # Aggregate tool call arguments across chunks
                # Note: The type: ignore comments below suppress pre-existing mypy
                # issues that surfaced when we added the
                # chunk: OpenAIChatCompletionChunk annotation above.
                if chunk_choice.delta.tool_calls:
                    for tool_call in chunk_choice.delta.tool_calls:
                        response_tool_call = chat_response_tool_calls.get(tool_call.index, None)  # type: ignore[arg-type]
                        # Create new tool call entry if this is the first chunk for this index
                        is_new_tool_call = response_tool_call is None
                        if is_new_tool_call:
                            tool_call_dict: dict[str, Any] = tool_call.model_dump()
                            tool_call_dict.pop("type", None)
                            # arguments may be None in the first streaming delta (name/index arrive before arguments)
                            # Initialize to "" so subsequent argument chunks accumulate correctly.
                            # The final "{}" fallback is applied at the end of streaming.
                            if tool_call_dict.get("function") and tool_call_dict["function"].get("arguments") is None:
                                tool_call_dict["function"]["arguments"] = ""
                            response_tool_call = OpenAIChatCompletionToolCall(**tool_call_dict)
                            chat_response_tool_calls[tool_call.index] = response_tool_call  # type: ignore[index]

                            # Create item ID for this tool call for streaming events
                            tool_call_item_id = f"fc_{uuid.uuid4()}"
                            tool_call_item_ids[tool_call.index] = tool_call_item_id  # type: ignore[index]

                            # Emit output_item.added event for the new function call
                            self.sequence_number += 1
                            is_mcp_tool = tool_call.function.name and tool_call.function.name in self.mcp_tool_to_server  # type: ignore[union-attr]
                            if not is_mcp_tool and tool_call.function.name not in _SERVER_SIDE_BUILTIN_TOOL_NAMES:  # type: ignore[union-attr]
                                # for MCP tools (and even other non-function tools) we emit an output message item later
                                function_call_item = OpenAIResponseOutputMessageFunctionToolCall(
                                    arguments="",  # Will be filled incrementally via delta events
                                    call_id=tool_call.id or "",
                                    name=tool_call.function.name if tool_call.function else "",
                                    id=tool_call_item_id,
                                    status="in_progress",
                                )
                                yield OpenAIResponseObjectStreamResponseOutputItemAdded(
                                    response_id=self.response_id,
                                    item=function_call_item,
                                    output_index=len(output_messages),
                                    sequence_number=self.sequence_number,
                                )

                        # Stream tool call arguments as they arrive (differentiate between MCP and function calls)
                        if tool_call.function and tool_call.function.arguments:
                            tool_call_item_id = tool_call_item_ids[tool_call.index]  # type: ignore[index]
                            self.sequence_number += 1

                            # Check if this is an MCP tool call
                            is_mcp_tool = tool_call.function.name and tool_call.function.name in self.mcp_tool_to_server
                            if is_mcp_tool:
                                # Emit MCP-specific argument delta event
                                yield OpenAIResponseObjectStreamResponseMcpCallArgumentsDelta(
                                    delta=tool_call.function.arguments,
                                    item_id=tool_call_item_id,
                                    output_index=len(output_messages),
                                    sequence_number=self.sequence_number,
                                )
                            else:
                                # Emit function call argument delta event
                                yield OpenAIResponseObjectStreamResponseFunctionCallArgumentsDelta(
                                    delta=tool_call.function.arguments,
                                    item_id=tool_call_item_id,
                                    output_index=len(output_messages),
                                    sequence_number=self.sequence_number,
                                )

                            # Accumulate arguments for final response (only for subsequent chunks)
                            if not is_new_tool_call and response_tool_call is not None:
                                # Both should have functions since we're inside the tool_call.function check above
                                assert response_tool_call.function is not None
                                assert tool_call.function is not None
                                response_tool_call.function.arguments = (
                                    response_tool_call.function.arguments or ""
                                ) + tool_call.function.arguments

            # Output Safety Validation for this chunk
            if self.guardrail_ids:
                # Check guardrails on accumulated text so far
                accumulated_text = "".join(chat_response_content)
                violation_message = await run_guardrails(self.safety_api, accumulated_text, self.guardrail_ids)
                if violation_message:
                    logger.info("Output guardrail violation", violation_message=violation_message)
                    chunk_events.clear()
                    yield await self._create_refusal_response(violation_message)
                    self.violation_detected = True
                    return
                else:
                    # No violation detected, emit all content events for this chunk
                    for event in chunk_events:
                        yield event

        # Emit arguments.done events for completed tool calls (differentiate between MCP and function calls)
        for tool_call_index in sorted(chat_response_tool_calls.keys()):
            tool_call = chat_response_tool_calls[tool_call_index]
            # Ensure that arguments, if sent back to the inference provider, are not None
            if tool_call.function:
                tool_call.function.arguments = tool_call.function.arguments or "{}"
            tool_call_item_id = tool_call_item_ids[tool_call_index]
            final_arguments: str = tool_call.function.arguments or "{}" if tool_call.function else "{}"
            func = chat_response_tool_calls[tool_call_index].function

            tool_call_name = func.name if func else ""

            # Check if this is an MCP tool call
            is_mcp_tool = tool_call_name and tool_call_name in self.mcp_tool_to_server
            self.sequence_number += 1
            done_event_cls = (
                OpenAIResponseObjectStreamResponseMcpCallArgumentsDone
                if is_mcp_tool
                else OpenAIResponseObjectStreamResponseFunctionCallArgumentsDone
            )
            yield done_event_cls(
                arguments=final_arguments,
                item_id=tool_call_item_id,
                output_index=len(output_messages),
                sequence_number=self.sequence_number,
            )

        # Emit content_part.done event if text content was streamed (before content gets cleared)
        if content_part_emitted:
            final_text = "".join(chat_response_content)
            self.sequence_number += 1
            yield OpenAIResponseObjectStreamResponseContentPartDone(
                content_index=content_index,
                response_id=self.response_id,
                item_id=message_item_id,
                output_index=message_output_index,
                part=OpenAIResponseContentPartOutputText(
                    text=final_text,
                    logprobs=[],
                ),
                sequence_number=self.sequence_number,
            )

        # Emit reasoning done events if reasoning content was streamed
        if reasoning_part_emitted:
            async for event in self._emit_reasoning_done_events(
                reasoning_text_accumulated=reasoning_text_accumulated,
                reasoning_content_index=reasoning_content_index,
                message_item_id=message_item_id,
                message_output_index=message_output_index,
            ):
                yield event

        # Emit refusal done events if refusal content was streamed
        if refusal_part_emitted:
            async for event in self._emit_refusal_done_events(
                refusal_text_accumulated=refusal_text_accumulated,
                refusal_content_index=refusal_content_index,
                message_item_id=message_item_id,
                message_output_index=message_output_index,
            ):
                yield event

        # Clear content when there are tool calls (OpenAI spec behavior)
        if chat_response_tool_calls:
            chat_response_content = []

        # Emit output_item.done for message when we have content and no tool calls
        if message_item_added_emitted and not chat_response_tool_calls:
            content_parts = []
            if content_part_emitted:
                final_text = "".join(chat_response_content)
                content_parts.append(
                    OpenAIResponseOutputMessageContentOutputText(
                        text=final_text,
                        annotations=[],
                        logprobs=chat_response_logprobs if chat_response_logprobs else [],
                    )
                )

            self.sequence_number += 1
            message_item = OpenAIResponseMessage(
                id=message_item_id,
                content=content_parts,
                role="assistant",
                status="completed",
            )
            yield OpenAIResponseObjectStreamResponseOutputItemDone(
                response_id=self.response_id,
                item=message_item,
                output_index=message_output_index,
                sequence_number=self.sequence_number,
            )

        yield ChatCompletionResult(
            response_id=chat_response_id,
            content=chat_response_content,
            tool_calls=chat_response_tool_calls,
            created=chunk_created,
            model=chunk_model,
            finish_reason=chunk_finish_reason,
            message_item_id=message_item_id,
            tool_call_item_ids=tool_call_item_ids,
            content_part_emitted=content_part_emitted,
            logprobs=chat_response_logprobs if chat_response_logprobs else None,
            service_tier=chunk_service_tier,
            reasoning_content="".join(reasoning_text_accumulated) if reasoning_text_accumulated else None,
        )

    def _build_chat_completion(self, result: ChatCompletionResult) -> OpenAIChatCompletion:
        """Build OpenAIChatCompletion from ChatCompletionResult."""
        # Convert collected chunks to complete response
        tool_calls: list[OpenAIChatCompletionToolCall | OpenAIChatCompletionCustomToolCall] | None
        if result.tool_calls:
            tool_calls = [result.tool_calls[i] for i in sorted(result.tool_calls.keys())]
        else:
            tool_calls = None

        assistant_message = OpenAIChatCompletionResponseMessage(
            content=result.content_text,
            tool_calls=tool_calls,
        )
        return OpenAIChatCompletion(
            id=result.response_id,
            choices=[
                OpenAIChoice(
                    message=assistant_message,
                    finish_reason=result.finish_reason,
                    index=0,
                    logprobs=OpenAIChoiceLogprobs(content=result.logprobs) if result.logprobs else None,
                )
            ],
            created=result.created,
            model=result.model,
        )

    async def _coordinate_tool_execution(
        self,
        function_tool_calls: list,
        non_function_tool_calls: list,
        completion_result_data: ChatCompletionResult,
        output_messages: list[OpenAIResponseOutput],
        next_turn_messages: list,
    ) -> AsyncIterator[OpenAIResponseObjectStream]:
        """Coordinate execution of both function and non-function tool calls."""
        # Execute non-function tool calls
        for tool_call in non_function_tool_calls:
            # if total calls made to built-in and mcp tools exceed max_tool_calls
            # then create a tool response message indicating the call was skipped
            if self.max_tool_calls is not None and self.accumulated_builtin_tool_calls >= self.max_tool_calls:
                logger.info(
                    "Ignoring built-in and mcp tool call since reached the limit", max_tool_calls=self.max_tool_calls
                )
                skipped_call_message = OpenAIToolMessageParam(
                    content=f"Tool call skipped: maximum tool calls limit ({self.max_tool_calls}) reached.",
                    tool_call_id=tool_call.id,
                )
                next_turn_messages.append(skipped_call_message)
                continue

            # Find the item_id for this tool call
            matching_item_id = None
            for index, item_id in completion_result_data.tool_call_item_ids.items():
                response_tool_call = completion_result_data.tool_calls.get(index)
                if response_tool_call and response_tool_call.id == tool_call.id:
                    matching_item_id = item_id
                    break

            # Use a fallback item_id if not found
            if not matching_item_id:
                matching_item_id = f"tc_{uuid.uuid4()}"

            self.sequence_number += 1
            if tool_call.function.name and tool_call.function.name in self.mcp_tool_to_server:
                item: OpenAIResponseOutput = OpenAIResponseOutputMessageMCPCall(
                    arguments="",
                    name=tool_call.function.name,
                    id=matching_item_id,
                    server_label=self.mcp_tool_to_server[tool_call.function.name].server_label,
                )
            elif tool_call.function.name == "web_search":
                item = OpenAIResponseOutputMessageWebSearchToolCall(
                    id=matching_item_id,
                    status="in_progress",
                )
            elif tool_call.function.name in ("knowledge_search", "file_search"):
                item = OpenAIResponseOutputMessageFileSearchToolCall(
                    id=matching_item_id,
                    status="in_progress",
                    queries=[tool_call.function.arguments or ""],
                )
            else:
                raise ValueError(f"Unsupported tool call: {tool_call.function.name}")

            yield OpenAIResponseObjectStreamResponseOutputItemAdded(
                response_id=self.response_id,
                item=item,
                output_index=len(output_messages),
                sequence_number=self.sequence_number,
            )

            # Execute tool call with streaming
            tool_call_log = None
            tool_response_message = None
            async for result in self.tool_executor.execute_tool_call(
                tool_call,
                self.ctx,
                self.sequence_number,
                len(output_messages),
                matching_item_id,
                self.mcp_tool_to_server,
            ):
                if result.stream_event:
                    # Forward streaming events
                    self.sequence_number = result.sequence_number
                    yield result.stream_event

                if result.final_output_message is not None:
                    tool_call_log = result.final_output_message
                    tool_response_message = result.final_input_message
                    self.sequence_number = result.sequence_number
                    if result.citation_files:
                        self.citation_files.update(result.citation_files)

            if tool_call_log:
                output_messages.append(tool_call_log)

                # Emit output_item.done event for completed non-function tool call
                if matching_item_id:
                    self.sequence_number += 1
                    yield OpenAIResponseObjectStreamResponseOutputItemDone(
                        response_id=self.response_id,
                        item=tool_call_log,
                        output_index=len(output_messages) - 1,
                        sequence_number=self.sequence_number,
                    )

            if tool_response_message:
                next_turn_messages.append(tool_response_message)

            # Track number of calls made to built-in and mcp tools
            self.accumulated_builtin_tool_calls += 1

        # Execute function tool calls (client-side)
        for tool_call in function_tool_calls:
            # Find the item_id for this tool call from our tracking dictionary
            matching_item_id = None
            for index, item_id in completion_result_data.tool_call_item_ids.items():
                response_tool_call = completion_result_data.tool_calls.get(index)
                if response_tool_call and response_tool_call.id == tool_call.id:
                    matching_item_id = item_id
                    break

            # Use existing item_id or create new one if not found
            final_item_id = matching_item_id or f"fc_{uuid.uuid4()}"

            function_call_item = OpenAIResponseOutputMessageFunctionToolCall(
                arguments=tool_call.function.arguments or "",
                call_id=tool_call.id,
                name=tool_call.function.name or "",
                id=final_item_id,
                status="completed",
            )
            output_messages.append(function_call_item)

            # Emit output_item.done event for completed function call
            self.sequence_number += 1
            yield OpenAIResponseObjectStreamResponseOutputItemDone(
                response_id=self.response_id,
                item=function_call_item,
                output_index=len(output_messages) - 1,
                sequence_number=self.sequence_number,
            )

    async def _process_new_tools(
        self, tools: list[OpenAIResponseInputTool], output_messages: list[OpenAIResponseOutput]
    ) -> AsyncIterator[OpenAIResponseObjectStream]:
        """Process all tools and emit appropriate streaming events."""

        def make_openai_tool(tool_name: str, tool: ToolDef) -> ChatCompletionToolParam:
            return convert_tooldef_to_openai_tool(
                tool_name=tool_name,
                description=tool.description,
                input_schema=tool.input_schema,
            )  # type: ignore[return-value]  # Returns dict but ChatCompletionToolParam expects TypedDict

        # Initialize chat_tools if not already set
        if self.ctx.chat_tools is None:
            self.ctx.chat_tools = []

        for input_tool in tools:
            if input_tool.type == "function":
                self.ctx.chat_tools.append(
                    ChatCompletionToolParam(type="function", function=input_tool.model_dump(exclude_none=True))  # type: ignore[typeddict-item,arg-type]  # Dict compatible with FunctionDefinition
                )
            elif input_tool.type in WebSearchToolTypes:
                tool_name = "web_search"
                # Need to access tool_groups_api from tool_executor
                tool = await self.tool_executor.tool_groups_api.get_tool(tool_name)
                if not tool:
                    raise ValueError(f"Tool {tool_name} not found")
                self.ctx.chat_tools.append(make_openai_tool(tool_name, tool))
            elif input_tool.type == "file_search":
                tool_name = "file_search"
                file_search_tool_def = ToolDef(
                    name=tool_name,
                    description="Search files for relevant information",
                    input_schema={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The search query",
                            },
                        },
                        "required": ["query"],
                    },
                )
                self.ctx.chat_tools.append(make_openai_tool(tool_name, file_search_tool_def))
            elif input_tool.type == "mcp":
                async for stream_event in self._process_mcp_tool(input_tool, output_messages):
                    yield stream_event
            else:
                raise ValueError(f"OGX OpenAI Responses does not yet support tool type: {input_tool.type}")

    async def _process_mcp_tool(
        self, mcp_tool: OpenAIResponseInputToolMCP, output_messages: list[OpenAIResponseOutput]
    ) -> AsyncIterator[OpenAIResponseObjectStream]:
        """Process an MCP tool configuration and emit appropriate streaming events."""
        # Resolve connector_id to server_url if provided
        if self.connectors_api is not None:
            mcp_tool = await resolve_mcp_connector_id(mcp_tool, self.connectors_api)

        # Emit mcp_list_tools.in_progress
        self.sequence_number += 1
        yield OpenAIResponseObjectStreamResponseMcpListToolsInProgress(
            sequence_number=self.sequence_number,
        )
        try:
            # Parse allowed/never allowed tools
            always_allowed = None
            never_allowed = None
            if mcp_tool.allowed_tools:
                if isinstance(mcp_tool.allowed_tools, list):
                    always_allowed = mcp_tool.allowed_tools
                elif isinstance(mcp_tool.allowed_tools, AllowedToolsFilter):
                    # AllowedToolsFilter only has tool_names field (not allowed/disallowed)
                    always_allowed = mcp_tool.allowed_tools.tool_names

            # Call list_mcp_tools
            tool_defs = None
            list_id = f"mcp_list_{uuid.uuid4()}"

            # Get session manager from tool_executor if available (fix for #4452)
            session_manager = getattr(self.tool_executor, "mcp_session_manager", None)

            if not mcp_tool.server_url:
                raise ValueError(
                    f"Failed to list MCP tools for server '{mcp_tool.server_label}': server_url is not set"
                )

            attributes = {
                "server_label": mcp_tool.server_label,
                "server_url": mcp_tool.server_url,
                "mcp_list_tools_id": list_id,
            }

            # TODO: follow semantic conventions for Open Telemetry tool spans
            # https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-spans/#execute-tool-span
            with tracer.start_as_current_span("list_mcp_tools", attributes=attributes):
                tool_defs = await list_mcp_tools(
                    endpoint=mcp_tool.server_url,
                    headers=mcp_tool.headers,
                    authorization=mcp_tool.authorization,
                    session_manager=session_manager,
                )

            # Create the MCP list tools message
            mcp_list_message = OpenAIResponseOutputMessageMCPListTools(
                id=list_id,
                server_label=mcp_tool.server_label,
                tools=[],
            )

            # Process tools and update context
            for t in tool_defs.data:
                if never_allowed and t.name in never_allowed:
                    continue
                if not always_allowed or t.name in always_allowed:
                    # Add to chat tools for inference
                    openai_tool = convert_tooldef_to_chat_tool(t)
                    if self.ctx.chat_tools is None:
                        self.ctx.chat_tools = []
                    self.ctx.chat_tools.append(openai_tool)  # type: ignore[arg-type]  # Returns dict but ChatCompletionToolParam expects TypedDict

                    # Add to MCP tool mapping
                    if t.name in self.mcp_tool_to_server:
                        raise ValueError(f"Duplicate tool name {t.name} found for server {mcp_tool.server_label}")
                    self.mcp_tool_to_server[t.name] = mcp_tool

                    # Add to reverse mapping for efficient server_label lookup
                    if mcp_tool.server_label not in self.server_label_to_tools:
                        self.server_label_to_tools[mcp_tool.server_label] = []
                    self.server_label_to_tools[mcp_tool.server_label].append(t.name)

                    # Add to MCP list message
                    mcp_list_message.tools.append(
                        MCPListToolsTool(
                            name=t.name,
                            description=t.description,
                            input_schema=t.input_schema
                            or {
                                "type": "object",
                                "properties": {},
                                "required": [],
                            },
                        )
                    )
            async for stream_event in self._add_mcp_list_tools(mcp_list_message, output_messages):
                yield stream_event

        except Exception as e:
            # TODO: Emit mcp_list_tools.failed event if needed
            logger.exception("Failed to list MCP tools", server_url=mcp_tool.server_url, error=str(e))
            raise

    async def _process_tools(
        self, output_messages: list[OpenAIResponseOutput]
    ) -> AsyncIterator[OpenAIResponseObjectStream]:
        # Handle all mcp tool lists from previous response that are still valid:
        # tool_context can be None when no tools are provided in the response request
        if self.ctx.tool_context:
            for tool in self.ctx.tool_context.previous_tool_listings:
                async for evt in self._reuse_mcp_list_tools(tool, output_messages):
                    yield evt
            # Process all remaining tools (including MCP tools) and emit streaming events
            if self.ctx.tool_context.tools_to_process:
                async for stream_event in self._process_new_tools(
                    self.ctx.tool_context.tools_to_process, output_messages
                ):
                    yield stream_event

    def _approval_required(self, tool_name: str) -> bool:
        if tool_name not in self.mcp_tool_to_server:
            return False
        mcp_server = self.mcp_tool_to_server[tool_name]
        if mcp_server.require_approval == "always":
            return True
        if mcp_server.require_approval == "never":
            return False
        if isinstance(mcp_server.require_approval, ApprovalFilter):
            if mcp_server.require_approval.always and tool_name in mcp_server.require_approval.always:
                return True
            if mcp_server.require_approval.never and tool_name in mcp_server.require_approval.never:
                return False
        return True

    async def _add_mcp_approval_request(
        self, tool_name: str, arguments: str, output_messages: list[OpenAIResponseOutput]
    ) -> AsyncIterator[OpenAIResponseObjectStream]:
        mcp_server = self.mcp_tool_to_server[tool_name]
        mcp_approval_request = OpenAIResponseMCPApprovalRequest(
            arguments=arguments,
            id=f"approval_{uuid.uuid4()}",
            name=tool_name,
            server_label=mcp_server.server_label,
        )
        output_messages.append(mcp_approval_request)

        self.sequence_number += 1
        yield OpenAIResponseObjectStreamResponseOutputItemAdded(
            response_id=self.response_id,
            item=mcp_approval_request,
            output_index=len(output_messages) - 1,
            sequence_number=self.sequence_number,
        )
        self.sequence_number += 1
        yield OpenAIResponseObjectStreamResponseOutputItemDone(
            response_id=self.response_id,
            item=mcp_approval_request,
            output_index=len(output_messages) - 1,
            sequence_number=self.sequence_number,
        )

    async def _add_mcp_list_tools(
        self, mcp_list_message: OpenAIResponseOutputMessageMCPListTools, output_messages: list[OpenAIResponseOutput]
    ) -> AsyncIterator[OpenAIResponseObjectStream]:
        # Add the MCP list message to output
        output_messages.append(mcp_list_message)

        # Emit output_item.added for the MCP list tools message
        self.sequence_number += 1
        yield OpenAIResponseObjectStreamResponseOutputItemAdded(
            response_id=self.response_id,
            item=OpenAIResponseOutputMessageMCPListTools(
                id=mcp_list_message.id,
                server_label=mcp_list_message.server_label,
                tools=[],
            ),
            output_index=len(output_messages) - 1,
            sequence_number=self.sequence_number,
        )
        # Emit mcp_list_tools.completed
        self.sequence_number += 1
        yield OpenAIResponseObjectStreamResponseMcpListToolsCompleted(
            sequence_number=self.sequence_number,
        )

        # Emit output_item.done for the MCP list tools message
        self.sequence_number += 1
        yield OpenAIResponseObjectStreamResponseOutputItemDone(
            response_id=self.response_id,
            item=mcp_list_message,
            output_index=len(output_messages) - 1,
            sequence_number=self.sequence_number,
        )

    async def _reuse_mcp_list_tools(
        self, original: OpenAIResponseOutputMessageMCPListTools, output_messages: list[OpenAIResponseOutput]
    ) -> AsyncIterator[OpenAIResponseObjectStream]:
        for t in original.tools:
            openai_tool = convert_tooldef_to_openai_tool(
                tool_name=t.name,
                description=t.description,
                input_schema=t.input_schema,
            )
            if self.ctx.chat_tools is None:
                self.ctx.chat_tools = []
            self.ctx.chat_tools.append(openai_tool)  # type: ignore[arg-type]  # Returns dict but ChatCompletionToolParam expects TypedDict

        mcp_list_message = OpenAIResponseOutputMessageMCPListTools(
            id=f"mcp_list_{uuid.uuid4()}",
            server_label=original.server_label,
            tools=original.tools,
        )

        async for stream_event in self._add_mcp_list_tools(mcp_list_message, output_messages):
            yield stream_event


async def _process_tool_choice(
    chat_tools: list[ChatCompletionToolParam],
    tool_choice: OpenAIResponseInputToolChoice,
    server_label_to_tools: dict[str, list[str]],
) -> str | OpenAIChatCompletionToolChoice | None:
    """Process and validate the OpenAI Responses tool choice and return the appropriate chat completion tool choice object.

    :param chat_tools: The list of chat tools to enforce tool choice against.
    :param tool_choice: The OpenAI Responses tool choice to process.
    :param server_label_to_tools: A dictionary mapping server labels to the list of tools available on that server.
    :return: The appropriate chat completion tool choice object.
    """

    # retrieve all function tool names from the chat tools
    # Note: chat_tools contains dicts, not objects
    chat_tool_names = [tool["function"]["name"] for tool in chat_tools if tool["type"] == "function"]

    if isinstance(tool_choice, OpenAIResponseInputToolChoiceMode):
        if tool_choice.value == "required":
            if len(chat_tool_names) == 0:
                return None

            # add all function tools to the allowed tools list and set mode to required
            return OpenAIChatCompletionToolChoiceAllowedTools(
                tools=[{"type": "function", "function": {"name": tool}} for tool in chat_tool_names],
                mode="required",
            )
        # return other modes as is
        return tool_choice.value

    elif isinstance(tool_choice, OpenAIResponseInputToolChoiceAllowedTools):
        # ensure that specified tool choices are available in the chat tools, if not, remove them from the list
        final_tools: list[dict[str, Any]] = []
        for tool in tool_choice.tools:
            match tool.get("type"):
                case "function":
                    final_tools.append({"type": "function", "function": {"name": tool.get("name")}})
                case "custom":
                    final_tools.append({"type": "custom", "custom": {"name": tool.get("name")}})
                case "mcp":
                    mcp_tools = convert_mcp_tool_choice(
                        chat_tool_names, tool.get("server_label"), server_label_to_tools, None
                    )
                    # convert_mcp_tool_choice can return a dict, list, or None
                    if isinstance(mcp_tools, list):
                        final_tools.extend(mcp_tools)
                    elif isinstance(mcp_tools, dict):
                        final_tools.append(mcp_tools)
                    # Skip if None or empty
                case "file_search":
                    final_tools.append({"type": "function", "function": {"name": "file_search"}})
                case _ if tool["type"] in WebSearchToolTypes:
                    final_tools.append({"type": "function", "function": {"name": "web_search"}})
                case _:
                    logger.warning("Unsupported tool type, skipping tool choice enforcement", tool_type=tool["type"])
                    continue

        return OpenAIChatCompletionToolChoiceAllowedTools(
            tools=final_tools,
            mode=tool_choice.mode,
        )

    else:
        # Handle specific tool choice by type
        # Each case validates the tool exists in chat_tools before returning
        match tool_choice:
            case OpenAIResponseInputToolChoiceCustomTool():
                if tool_choice.name and tool_choice.name not in chat_tool_names:
                    logger.warning("Tool not found in chat tools", tool_name=tool_choice.name)
                    return None
                return OpenAIChatCompletionToolChoiceCustomTool(name=tool_choice.name)

            case OpenAIResponseInputToolChoiceFunctionTool():
                if tool_choice.name and tool_choice.name not in chat_tool_names:
                    logger.warning("Tool not found in chat tools", tool_name=tool_choice.name)
                    return None
                return OpenAIChatCompletionToolChoiceFunctionTool(name=tool_choice.name)

            case OpenAIResponseInputToolChoiceFileSearch():
                if "file_search" not in chat_tool_names:
                    logger.warning("Tool file_search not found in chat tools")
                    return None
                return OpenAIChatCompletionToolChoiceFunctionTool(name="file_search")

            case OpenAIResponseInputToolChoiceWebSearch():
                if "web_search" not in chat_tool_names:
                    logger.warning("Tool web_search not found in chat tools")
                    return None
                return OpenAIChatCompletionToolChoiceFunctionTool(name="web_search")

            case OpenAIResponseInputToolChoiceMCPTool():
                mcp_result = convert_mcp_tool_choice(
                    chat_tool_names,
                    tool_choice.server_label,
                    server_label_to_tools,
                    tool_choice.name,
                )
                if isinstance(mcp_result, dict):
                    # for single tool choice, return as function tool choice
                    function_info = mcp_result["function"]
                    if not isinstance(function_info, dict):
                        return None
                    return OpenAIChatCompletionToolChoiceFunctionTool(name=function_info["name"])
                elif isinstance(mcp_result, list):
                    # for multiple tool choices, return as allowed tools
                    return OpenAIChatCompletionToolChoiceAllowedTools(
                        tools=mcp_result,
                        mode="required",
                    )
                return None


async def resolve_mcp_connector_id(
    mcp_tool: OpenAIResponseInputToolMCP,
    connectors_api: Connectors,
) -> OpenAIResponseInputToolMCP:
    """Resolve connector_id to server_url for an MCP tool.

    If the mcp_tool has a connector_id but no server_url, this function
    looks up the connector and populates the server_url from it.

    Args:
        mcp_tool: The MCP tool configuration to resolve
        connectors_api: The connectors API for looking up connectors

    Returns:
        The mcp_tool with server_url populated (may be same instance if already set)
    """
    if mcp_tool.connector_id and not mcp_tool.server_url:
        connector = await connectors_api.get_connector(GetConnectorRequest(connector_id=mcp_tool.connector_id))
        return mcp_tool.model_copy(update={"server_url": connector.url})
    return mcp_tool
