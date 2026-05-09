# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import asyncio
import re
import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

import tiktoken
from pydantic import BaseModel, TypeAdapter

from ogx.core.conversations.validation import CONVERSATION_ID_PATTERN
from ogx.core.task import (
    RequestContext,
    activate_request_context,
    capture_request_context,
    create_detached_background_task,
)
from ogx.log import get_logger
from ogx.providers.inline.responses.builtin.config import CompactionConfig
from ogx.providers.utils.responses.responses_store import (
    ResponsesStore,
    _OpenAIResponseObjectWithInputAndMessages,
)
from ogx.providers.utils.tools.mcp import MCPSessionManager
from ogx_api import (
    AddItemsRequest,
    ConflictError,
    Connectors,
    ConversationItem,
    Conversations,
    Files,
    GetPromptRequest,
    Inference,
    InternalServerError,
    InvalidParameterError,
    ListItemsRequest,
    ListOpenAIResponseInputItem,
    ListOpenAIResponseObject,
    OpenAIChatCompletionContentPartParam,
    OpenAICompactedResponse,
    OpenAIDeleteResponseObject,
    OpenAIMessageParam,
    OpenAIResponseCompaction,
    OpenAIResponseError,
    OpenAIResponseInput,
    OpenAIResponseInputMessageContentFile,
    OpenAIResponseInputMessageContentImage,
    OpenAIResponseInputMessageContentText,
    OpenAIResponseInputTool,
    OpenAIResponseInputToolChoice,
    OpenAIResponseMessage,
    OpenAIResponseObject,
    OpenAIResponseObjectStream,
    OpenAIResponsePrompt,
    OpenAIResponseReasoning,
    OpenAIResponseText,
    OpenAIResponseTextFormat,
    OpenAIResponseUsage,
    OpenAIResponseUsageInputTokensDetails,
    OpenAIResponseUsageOutputTokensDetails,
    OpenAISystemMessageParam,
    OpenAIUserMessageParam,
    Order,
    Prompts,
    ResponseGuardrailSpec,
    ResponseItemInclude,
    ResponseStreamOptions,
    ResponseTruncation,
    Safety,
    ServiceNotEnabledError,
    ToolGroups,
    ToolRuntime,
    VectorIO,
)
from ogx_api.inference import OpenAIChatCompletionRequestWithExtraBody, ServiceTier

from .streaming import StreamingResponseOrchestrator
from .tool_executor import ToolExecutor
from .types import ChatCompletionContext, ToolContext
from .utils import (
    convert_response_content_to_chat_content,
    convert_response_input_to_chat_messages,
    convert_response_text_to_chat_response_format,
    extract_guardrail_ids,
)

logger = get_logger(name=__name__, category="openai_responses")

BACKGROUND_RESPONSE_TIMEOUT_SECONDS = 300  # 5 minutes
BACKGROUND_QUEUE_MAX_SIZE = 100
BACKGROUND_NUM_WORKERS = 10


@dataclass
class _BackgroundWorkItem:
    """Typed queue item that pairs business kwargs with the originating request context."""

    request_context: RequestContext
    kwargs: dict = field(default_factory=dict)


class OpenAIResponsePreviousResponseWithInputItems(BaseModel):
    """Container for a previous response object paired with its input items."""

    input_items: ListOpenAIResponseInputItem
    response: OpenAIResponseObject


class OpenAIResponsesImpl:
    """Implementation of the OpenAI Responses API with streaming, tool calling, and persistence."""

    def __init__(
        self,
        inference_api: Inference,
        tool_groups_api: ToolGroups,
        tool_runtime_api: ToolRuntime,
        responses_store: ResponsesStore,
        vector_io_api: VectorIO,  # VectorIO
        safety_api: Safety | None,
        conversations_api: Conversations,
        prompts_api: Prompts,
        files_api: Files,
        connectors_api: Connectors,
        vector_stores_config=None,
        compaction_config=None,
    ):
        self.inference_api = inference_api
        self.tool_groups_api = tool_groups_api
        self.tool_runtime_api = tool_runtime_api
        self.responses_store = responses_store
        self.vector_io_api = vector_io_api
        self.safety_api = safety_api
        self.conversations_api = conversations_api
        self.tool_executor = ToolExecutor(
            tool_groups_api=tool_groups_api,
            tool_runtime_api=tool_runtime_api,
            vector_io_api=vector_io_api,
            vector_stores_config=vector_stores_config,
        )
        self.prompts_api = prompts_api
        self.files_api = files_api
        self.connectors_api = connectors_api

        self.compaction_config = compaction_config or CompactionConfig()
        self._background_queue: asyncio.Queue[_BackgroundWorkItem] = asyncio.Queue(maxsize=BACKGROUND_QUEUE_MAX_SIZE)
        self._background_worker_tasks: set[asyncio.Task] = set()
        self._background_response_tasks: dict[str, asyncio.Task] = {}
        self._background_response_tasks_lock = asyncio.Lock()

    async def initialize(self) -> None:
        """No-op: background workers are started lazily on first use.

        Workers must be started in the event loop that handles requests, not the
        temporary loop used during provider initialization (which is destroyed after
        init completes, cancelling any tasks created there).
        """
        pass

    async def _ensure_workers_started(self) -> None:
        """Start background workers in the current event loop if not already running."""
        for _ in range(BACKGROUND_NUM_WORKERS - len(self._background_worker_tasks)):
            task = create_detached_background_task(self._background_worker())
            self._background_worker_tasks.add(task)
            task.add_done_callback(self._background_worker_tasks.discard)

    async def shutdown(self) -> None:
        """Stop background worker pool."""
        # Cancel all in-progress response tasks
        async with self._background_response_tasks_lock:
            for task in self._background_response_tasks.values():
                task.cancel()
            response_task_list = list(self._background_response_tasks.values())

        # Cancel worker tasks
        for task in self._background_worker_tasks:
            task.cancel()

        # Wait for all tasks to complete
        all_tasks = list(self._background_worker_tasks) + response_task_list
        await asyncio.gather(*all_tasks, return_exceptions=True)

    async def _background_worker(self) -> None:
        """Worker coroutine that pulls items from the queue and processes them."""
        while True:
            item = await self._background_queue.get()
            with activate_request_context(item.request_context):
                response_id = item.kwargs["response_id"]

                # Create a task for this specific response so we can cancel it
                processing_task = asyncio.create_task(
                    asyncio.wait_for(
                        self._run_background_response_loop(**item.kwargs),
                        timeout=BACKGROUND_RESPONSE_TIMEOUT_SECONDS,
                    )
                )

                # Track the task
                async with self._background_response_tasks_lock:
                    self._background_response_tasks[response_id] = processing_task

                try:
                    await processing_task
                except asyncio.CancelledError:
                    # Response was cancelled via cancel_openai_response
                    logger.info("Background response was cancelled", response_id=response_id)
                    try:
                        existing = await self.responses_store.get_response_object(response_id)
                        if existing.status != "cancelled":
                            existing.status = "cancelled"
                            await self.responses_store.update_response_object(existing)
                    except Exception:
                        logger.exception("Failed to update response with cancelled status", response_id=response_id)
                except TimeoutError:
                    logger.exception(
                        "Background response timed out",
                        response_id=response_id,
                        timeout_seconds=BACKGROUND_RESPONSE_TIMEOUT_SECONDS,
                    )
                    try:
                        existing = await self.responses_store.get_response_object(response_id)
                        existing.status = "failed"
                        existing.error = OpenAIResponseError(
                            code="processing_error",
                            message=f"Background response timed out after {BACKGROUND_RESPONSE_TIMEOUT_SECONDS}s",
                        )
                        await self.responses_store.update_response_object(existing)
                    except Exception:
                        logger.exception(
                            "Failed to update response with timeout status, client polling this response will not see the failure",
                            response_id=response_id,
                        )
                except Exception as e:
                    logger.exception("Failed to process background response", response_id=response_id)
                    try:
                        existing = await self.responses_store.get_response_object(response_id)
                        existing.status = "failed"
                        existing.error = OpenAIResponseError(
                            code="processing_error",
                            message=str(e),
                        )
                        await self.responses_store.update_response_object(existing)
                    except Exception:
                        logger.exception(
                            "Failed to update response with error status, client polling this response will not see the failure",
                            response_id=response_id,
                        )
                finally:
                    # Remove from tracking
                    async with self._background_response_tasks_lock:
                        self._background_response_tasks.pop(response_id, None)
                    self._background_queue.task_done()

    async def _prepend_previous_response(
        self,
        input: str | list[OpenAIResponseInput],
        previous_response: _OpenAIResponseObjectWithInputAndMessages,
    ):
        # Convert Sequence to list for mutation
        new_input_items = list(previous_response.input)
        new_input_items.extend(previous_response.output)

        if isinstance(input, str):
            new_input_items.append(OpenAIResponseMessage(content=input, role="user"))
        else:
            new_input_items.extend(input)

        return new_input_items

    async def _process_input_with_previous_response(
        self,
        input: str | list[OpenAIResponseInput],
        tools: list[OpenAIResponseInputTool] | None,
        previous_response_id: str | None,
        conversation: str | None,
    ) -> tuple[str | list[OpenAIResponseInput], list[OpenAIMessageParam], ToolContext, OpenAIResponseUsage | None]:
        """Process input with optional previous response context.

        Returns:
            tuple: (all_input for storage, messages for chat completion, tool context, previous usage)
        """
        tool_context = ToolContext(tools)
        previous_usage: OpenAIResponseUsage | None = None
        if previous_response_id:
            previous_response: _OpenAIResponseObjectWithInputAndMessages = (
                await self.responses_store.get_response_object(previous_response_id)
            )
            if previous_response.status in ("queued", "in_progress"):
                raise ValueError(
                    f"Response {previous_response_id} is still {previous_response.status}. "
                    "Cannot use an incomplete background response as previous_response_id."
                )
            previous_usage = previous_response.usage
            all_input = await self._prepend_previous_response(input, previous_response)

            if previous_response.messages:
                # Use stored messages directly and convert only new input
                message_adapter = TypeAdapter(list[OpenAIMessageParam])
                messages = message_adapter.validate_python(previous_response.messages)
                new_messages = await convert_response_input_to_chat_messages(
                    input, previous_messages=messages, files_api=self.files_api
                )
                messages.extend(new_messages)
            else:
                # Backward compatibility: reconstruct from inputs
                messages = await convert_response_input_to_chat_messages(all_input, files_api=self.files_api)

            tool_context.recover_tools_from_previous_response(previous_response)
        elif conversation is not None:
            conversation_items = await self.conversations_api.list_items(
                ListItemsRequest(conversation_id=conversation, order="asc")
            )

            # Use stored messages as source of truth (like previous_response.messages)
            stored_messages = await self.responses_store.get_conversation_messages(conversation)

            all_input = input
            if not conversation_items.data:
                # First turn - just convert the new input
                messages = await convert_response_input_to_chat_messages(input, files_api=self.files_api)
            else:
                if not stored_messages:
                    conv_items: list[OpenAIResponseInput] = list(conversation_items.data)
                    if isinstance(input, str):
                        conv_items.append(
                            OpenAIResponseMessage(
                                role="user", content=[OpenAIResponseInputMessageContentText(text=input)]
                            )
                        )
                    else:
                        conv_items.extend(input)
                    all_input = conv_items
                else:
                    all_input = input

                messages = stored_messages or []
                new_messages = await convert_response_input_to_chat_messages(
                    all_input, previous_messages=messages, files_api=self.files_api
                )
                messages.extend(new_messages)
        else:
            all_input = input
            messages = await convert_response_input_to_chat_messages(all_input, files_api=self.files_api)

        return all_input, messages, tool_context, previous_usage

    async def _prepend_prompt(
        self,
        messages: list[OpenAIMessageParam],
        openai_response_prompt: OpenAIResponsePrompt | None,
    ) -> None:
        """Prepend prompt template to messages, resolving text/image/file variables.

        :param messages: List of OpenAIMessageParam objects
        :param openai_response_prompt: (Optional) OpenAIResponsePrompt object with variables
        :returns: string of utf-8 characters
        """
        if not openai_response_prompt or not openai_response_prompt.id:
            return

        prompt_version = int(openai_response_prompt.version) if openai_response_prompt.version else None
        cur_prompt = await self.prompts_api.get_prompt(
            GetPromptRequest(prompt_id=openai_response_prompt.id, version=prompt_version)
        )

        if not cur_prompt or not cur_prompt.prompt:
            return

        cur_prompt_text = cur_prompt.prompt
        cur_prompt_variables = cur_prompt.variables

        if not openai_response_prompt.variables:
            messages.insert(0, OpenAISystemMessageParam(content=cur_prompt_text))
            return

        # Validate that all provided variables exist in the prompt
        for name in openai_response_prompt.variables.keys():
            if name not in cur_prompt_variables:
                raise InvalidParameterError(
                    "prompt.variables",
                    name,
                    f"Variable not defined in prompt '{openai_response_prompt.id}'.",
                )

        # Separate text and media variables
        text_substitutions = {}
        media_content_parts: list[OpenAIChatCompletionContentPartParam] = []

        for name, value in openai_response_prompt.variables.items():
            # Text variable found
            if isinstance(value, OpenAIResponseInputMessageContentText):
                text_substitutions[name] = value.text

            # Media variable found
            elif isinstance(value, OpenAIResponseInputMessageContentImage | OpenAIResponseInputMessageContentFile):
                converted_parts = await convert_response_content_to_chat_content([value], files_api=self.files_api)
                if isinstance(converted_parts, list):
                    media_content_parts.extend(converted_parts)

                # Eg: {{product_photo}} becomes "[Image: product_photo]"
                # This gives the model textual context about what media exists in the prompt
                var_type = value.type.replace("input_", "").replace("_", " ").title()
                text_substitutions[name] = f"[{var_type}: {name}]"

        def replace_variable(match: re.Match[str]) -> str:
            var_name = match.group(1).strip()
            return str(text_substitutions.get(var_name, match.group(0)))

        pattern = r"\{\{\s*(\w+)\s*\}\}"
        processed_prompt_text = re.sub(pattern, replace_variable, cur_prompt_text)

        # Insert system message with resolved text
        messages.insert(0, OpenAISystemMessageParam(content=processed_prompt_text))

        # If we have media, create a new user message because allows to ingest images and files
        if media_content_parts:
            messages.append(OpenAIUserMessageParam(content=media_content_parts))

    async def get_openai_response(
        self,
        response_id: str,
    ) -> OpenAIResponseObject:
        response_with_input = await self.responses_store.get_response_object(response_id)
        return response_with_input.to_response_object()

    async def list_openai_responses(
        self,
        after: str | None = None,
        limit: int | None = 50,
        model: str | None = None,
        order: Order | None = Order.desc,
    ) -> ListOpenAIResponseObject:
        return await self.responses_store.list_responses(after, limit, model, order)

    async def list_openai_response_input_items(
        self,
        response_id: str,
        after: str | None = None,
        before: str | None = None,
        include: list[ResponseItemInclude] | None = None,
        limit: int | None = 20,
        order: Order | None = Order.desc,
    ) -> ListOpenAIResponseInputItem:
        """List input items for a given OpenAI response.

        :param response_id: The ID of the response to retrieve input items for.
        :param after: An item ID to list items after, used for pagination.
        :param before: An item ID to list items before, used for pagination.
        :param include: Additional fields to include in the response.
        :param limit: A limit on the number of objects to be returned.
        :param order: The order to return the input items in.
        :returns: An ListOpenAIResponseInputItem.
        """
        return await self.responses_store.list_response_input_items(response_id, after, before, include, limit, order)

    async def _store_response(
        self,
        response: OpenAIResponseObject,
        input: str | list[OpenAIResponseInput],
        messages: list[OpenAIMessageParam],
    ) -> None:
        new_input_id = f"msg_{uuid.uuid4()}"
        # Type input_items_data as the full OpenAIResponseInput union to avoid list invariance issues
        input_items_data: list[OpenAIResponseInput] = []

        if isinstance(input, str):
            # synthesize a message from the input string
            input_content = OpenAIResponseInputMessageContentText(text=input)
            input_content_item = OpenAIResponseMessage(
                role="user",
                content=[input_content],
                id=new_input_id,
            )
            input_items_data = [input_content_item]
        else:
            # we already have a list of messages
            for input_item in input:
                if isinstance(input_item, OpenAIResponseMessage):
                    # These may or may not already have an id, so dump to dict, check for id, and add if missing
                    input_item_dict = input_item.model_dump()
                    if "id" not in input_item_dict:
                        input_item_dict["id"] = new_input_id
                    input_items_data.append(OpenAIResponseMessage(**input_item_dict))
                else:
                    input_items_data.append(input_item)

        await self.responses_store.store_response_object(
            response_object=response,
            input=input_items_data,
            messages=messages,
        )

    def _prepare_input_items_for_storage(
        self,
        input: str | list[OpenAIResponseInput],
    ) -> list[OpenAIResponseInput]:
        """Prepare input items for storage, adding IDs where needed.

        This method is called once at the start of streaming to prepare input items
        that will be reused across multiple persistence calls during streaming.
        """
        new_input_id = f"msg_{uuid.uuid4()}"
        input_items_data: list[OpenAIResponseInput] = []

        if isinstance(input, str):
            input_content = OpenAIResponseInputMessageContentText(text=input)
            input_content_item = OpenAIResponseMessage(
                role="user",
                content=[input_content],
                id=new_input_id,
            )
            input_items_data = [input_content_item]
        else:
            for input_item in input:
                if isinstance(input_item, OpenAIResponseMessage):
                    input_item_dict = input_item.model_dump()
                    if "id" not in input_item_dict:
                        input_item_dict["id"] = new_input_id
                    input_items_data.append(OpenAIResponseMessage(**input_item_dict))
                else:
                    input_items_data.append(input_item)

        return input_items_data

    async def _persist_streaming_state(
        self,
        stream_chunk: OpenAIResponseObjectStream,
        orchestrator,
        input_items: list[OpenAIResponseInput],
        output_items: list,
    ) -> None:
        """Persist response state at significant streaming events.

        This enables clients to poll GET /v1/responses/{response_id} during streaming
        to see in-progress turn state instead of empty results.

        Persistence occurs at:
        - response.in_progress: Initial INSERT with empty output
        - response.output_item.done: UPDATE with accumulated output items
        - response.completed/response.incomplete: Final UPDATE with complete state
        - response.failed: UPDATE with error state

        :param stream_chunk: The current streaming event.
        :param orchestrator: The streaming orchestrator (for snapshotting response).
        :param input_items: Pre-prepared input items for storage.
        :param output_items: Accumulated output items so far.
        """
        try:
            match stream_chunk.type:
                case "response.in_progress":
                    # Initial persistence when response starts
                    in_progress_response = stream_chunk.response
                    await self.responses_store.upsert_response_object(
                        response_object=in_progress_response,
                        input=input_items,
                        messages=[],
                    )

                case "response.output_item.done":
                    # Incremental update when an output item completes (tool call, message)
                    current_snapshot = orchestrator._snapshot_response(
                        status="in_progress",
                        outputs=output_items,
                    )
                    # Get current messages (filter out system messages)
                    messages_to_store = list(
                        filter(
                            lambda x: not isinstance(x, OpenAISystemMessageParam),
                            orchestrator.final_messages or orchestrator.ctx.messages,
                        )
                    )
                    await self.responses_store.upsert_response_object(
                        response_object=current_snapshot,
                        input=input_items,
                        messages=messages_to_store,
                    )

                case "response.completed" | "response.incomplete":
                    # Final persistence when response finishes
                    final_response = stream_chunk.response
                    messages_to_store = list(
                        filter(
                            lambda x: not isinstance(x, OpenAISystemMessageParam),
                            orchestrator.final_messages,
                        )
                    )
                    await self.responses_store.upsert_response_object(
                        response_object=final_response,
                        input=input_items,
                        messages=messages_to_store,
                    )

                case "response.failed":
                    # Persist failed state so GET shows error
                    failed_response = stream_chunk.response
                    # Preserve any accumulated non-system messages for failed responses
                    messages_to_store = list(
                        filter(
                            lambda x: not isinstance(x, OpenAISystemMessageParam),
                            orchestrator.final_messages or orchestrator.ctx.messages,
                        )
                    )
                    await self.responses_store.upsert_response_object(
                        response_object=failed_response,
                        input=input_items,
                        messages=messages_to_store,
                    )
        except Exception as e:
            # Best-effort persistence: log error but don't fail the stream
            logger.warning("Failed to persist streaming state", chunk_type=stream_chunk.type, error=str(e))

    async def create_openai_response(
        self,
        input: str | list[OpenAIResponseInput],
        model: str,
        background: bool | None = False,
        prompt: OpenAIResponsePrompt | None = None,
        instructions: str | None = None,
        previous_response_id: str | None = None,
        prompt_cache_key: str | None = None,
        conversation: str | None = None,
        store: bool | None = True,
        stream: bool | None = False,
        temperature: float | None = None,
        top_p: float | None = None,
        frequency_penalty: float | None = None,
        text: OpenAIResponseText | None = None,
        tool_choice: OpenAIResponseInputToolChoice | None = None,
        tools: list[OpenAIResponseInputTool] | None = None,
        include: list[ResponseItemInclude] | None = None,
        max_infer_iters: int | None = 10,
        guardrails: list[str | ResponseGuardrailSpec] | None = None,
        parallel_tool_calls: bool | None = None,
        max_tool_calls: int | None = None,
        reasoning: OpenAIResponseReasoning | None = None,
        max_output_tokens: int | None = None,
        safety_identifier: str | None = None,
        service_tier: ServiceTier | None = None,
        metadata: dict[str, str] | None = None,
        truncation: ResponseTruncation | None = None,
        top_logprobs: int | None = None,
        presence_penalty: float | None = None,
        extra_body: dict | None = None,
        stream_options: ResponseStreamOptions | None = None,
        context_management: list | None = None,
    ) -> OpenAIResponseObject | AsyncIterator[OpenAIResponseObjectStream]:
        stream = bool(stream)
        background = bool(background)
        text = OpenAIResponseText(format=OpenAIResponseTextFormat(type="text")) if text is None else text

        # Validate that stream and background are mutually exclusive
        if stream and background:
            raise ValueError("OGX does not yet support 'stream' and 'background' together.")

        if background and store is False:
            raise ValueError("Cannot use 'background' with 'store=False'. Background responses must be stored.")

        # Filter out unsupported include items instead of rejecting the request
        if include:
            include = [item for item in include if str(item) != "reasoning.encrypted_content"]

        # Validate MCP tools: ensure Authorization header is not passed via headers dict
        if tools:
            from ogx_api.openai_responses import OpenAIResponseInputToolMCP

            for tool in tools:
                if isinstance(tool, OpenAIResponseInputToolMCP) and tool.headers:
                    for key in tool.headers.keys():
                        if key.lower() == "authorization":
                            raise InvalidParameterError(
                                f"tools[server_label={tool.server_label!r}].headers",
                                key,
                                "Authorization credentials must be passed via the 'authorization' parameter, not 'headers'.",
                            )

        guardrail_ids = extract_guardrail_ids(guardrails) if guardrails else []

        # Validate that Safety API is available if guardrails are requested
        if guardrail_ids and self.safety_api is None:
            raise ServiceNotEnabledError(
                "Safety API",
                provider_specific_message="Ensure the Safety API is enabled in your stack, otherwise remove the 'guardrails' parameter from your request.",
            )

        if conversation is not None:
            if previous_response_id is not None:
                raise InvalidParameterError(
                    "previous_response_id, conversation",
                    "previous_response_id and conversation are both provided",
                    "Provide only one of these parameters.",
                )

            if not CONVERSATION_ID_PATTERN.fullmatch(conversation):
                raise InvalidParameterError(
                    "conversation",
                    conversation,
                    "Must match format 'conv_' followed by 48 lowercase hex characters.",
                )

        if max_tool_calls is not None and max_tool_calls < 1:
            raise ValueError(f"Invalid {max_tool_calls=}; should be >= 1")

        # Handle background mode
        if background:
            return await self._create_background_response(
                input=input,
                model=model,
                prompt=prompt,
                instructions=instructions,
                previous_response_id=previous_response_id,
                conversation=conversation,
                store=store,
                temperature=temperature,
                frequency_penalty=frequency_penalty,
                text=text,
                tool_choice=tool_choice,
                tools=tools,
                include=include,
                max_infer_iters=max_infer_iters,
                guardrail_ids=guardrail_ids,
                parallel_tool_calls=parallel_tool_calls,
                max_tool_calls=max_tool_calls,
                reasoning=reasoning,
                max_output_tokens=max_output_tokens,
                safety_identifier=safety_identifier,
                service_tier=service_tier,
                metadata=metadata,
                truncation=truncation,
                presence_penalty=presence_penalty,
                extra_body=extra_body,
                context_management=context_management,
            )

        stream_gen = self._create_streaming_response(
            input=input,
            conversation=conversation,
            model=model,
            prompt=prompt,
            instructions=instructions,
            previous_response_id=previous_response_id,
            prompt_cache_key=prompt_cache_key,
            store=store,
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            text=text,
            tools=tools,
            tool_choice=tool_choice,
            max_infer_iters=max_infer_iters,
            guardrail_ids=guardrail_ids,
            parallel_tool_calls=parallel_tool_calls,
            max_tool_calls=max_tool_calls,
            reasoning=reasoning,
            max_output_tokens=max_output_tokens,
            safety_identifier=safety_identifier,
            service_tier=service_tier,
            metadata=metadata,
            include=include,
            truncation=truncation,
            top_logprobs=top_logprobs,
            presence_penalty=presence_penalty,
            extra_body=extra_body,
            stream_options=stream_options,
            context_management=context_management,
        )

        if stream:
            return stream_gen
        else:
            final_response = None
            final_event_type = None
            failed_response = None
            async for stream_chunk in stream_gen:
                match stream_chunk.type:
                    case "response.completed" | "response.incomplete":
                        if final_response is not None:
                            logger.error(
                                "The response stream produced multiple terminal events, when it should produce exactly one",
                                response_id=stream_chunk.response.id,
                                first_terminal_event=final_event_type,
                                second_terminal_event=stream_chunk.type,
                                model=model,
                                conversation=conversation,
                                previous_response_id=previous_response_id,
                            )
                            raise InternalServerError()
                        final_response = stream_chunk.response
                        final_event_type = stream_chunk.type
                    case "response.failed":
                        failed_response = stream_chunk.response
                        error_message = (
                            failed_response.error.message
                            if failed_response.error
                            else "response failed but no error message was provided"
                        )
                        logger.error(
                            "response creation failed",
                            error_message=error_message,
                            response_id=failed_response.id,
                            model=model,
                        )
                        # Surface the provider message — it may be actionable (e.g. context window exceeded)
                        # and is already visible to callers in streaming mode via the response.failed event.
                        raise InternalServerError(error_message)
                    case _:
                        pass  # Other event types don't have .response

            if final_response is None:
                logger.error(
                    "The response stream never reached a terminal state",
                    model=model,
                    conversation=conversation,
                    previous_response_id=previous_response_id,
                )
                raise InternalServerError()
            # Preserve the request mode on the terminal response object returned to the caller.
            final_response.background = background
            return final_response

    async def _create_background_response(
        self,
        input: str | list[OpenAIResponseInput],
        model: str,
        prompt: OpenAIResponsePrompt | None = None,
        instructions: str | None = None,
        previous_response_id: str | None = None,
        conversation: str | None = None,
        store: bool | None = True,
        temperature: float | None = None,
        frequency_penalty: float | None = None,
        text: OpenAIResponseText | None = None,
        tool_choice: OpenAIResponseInputToolChoice | None = None,
        tools: list[OpenAIResponseInputTool] | None = None,
        include: list[ResponseItemInclude] | None = None,
        max_infer_iters: int | None = 10,
        guardrail_ids: list[str] | None = None,
        parallel_tool_calls: bool | None = None,
        max_tool_calls: int | None = None,
        reasoning: OpenAIResponseReasoning | None = None,
        max_output_tokens: int | None = None,
        safety_identifier: str | None = None,
        service_tier: ServiceTier | None = None,
        metadata: dict[str, str] | None = None,
        truncation: ResponseTruncation | None = None,
        presence_penalty: float | None = None,
        extra_body: dict | None = None,
        context_management: list | None = None,
    ) -> OpenAIResponseObject:
        """Create a response that processes in the background.

        Returns immediately with a queued response object.
        """
        # Start workers in the current (request-handling) event loop if not already running.
        await self._ensure_workers_started()

        response_id = f"resp_{uuid.uuid4()}"
        created_at = int(time.time())

        # Normalize input to list format for storage
        input_items: list[OpenAIResponseInput] = (
            [OpenAIResponseMessage(content=input, role="user")] if isinstance(input, str) else input
        )

        # Create initial queued response
        queued_response = OpenAIResponseObject(
            id=response_id,
            created_at=created_at,
            model=model,
            status="queued",
            output=[],
            background=True,
            parallel_tool_calls=parallel_tool_calls if parallel_tool_calls is not None else True,
            previous_response_id=previous_response_id,
            prompt=prompt,
            temperature=temperature,
            text=text if text else OpenAIResponseText(format=OpenAIResponseTextFormat(type="text")),
            tools=[],  # Will be populated when processing completes
            tool_choice=tool_choice,
            instructions=instructions,
            max_tool_calls=max_tool_calls,
            reasoning=reasoning,
            metadata=metadata,
            store=store if store is not None else True,
        )

        # Store the queued response
        await self.responses_store.store_response_object(
            response_object=queued_response,
            input=input_items,
            messages=[],
        )

        # Enqueue work item for background workers. Raises QueueFull if at capacity.
        try:
            self._background_queue.put_nowait(
                _BackgroundWorkItem(
                    request_context=capture_request_context(),
                    kwargs=dict(
                        response_id=response_id,
                        input=input,
                        model=model,
                        prompt=prompt,
                        instructions=instructions,
                        previous_response_id=previous_response_id,
                        conversation=conversation,
                        store=store,
                        temperature=temperature,
                        frequency_penalty=frequency_penalty,
                        text=text,
                        tool_choice=tool_choice,
                        tools=tools,
                        include=include,
                        max_infer_iters=max_infer_iters,
                        guardrail_ids=guardrail_ids,
                        parallel_tool_calls=parallel_tool_calls,
                        max_tool_calls=max_tool_calls,
                        reasoning=reasoning,
                        max_output_tokens=max_output_tokens,
                        safety_identifier=safety_identifier,
                        service_tier=service_tier,
                        metadata=metadata,
                        truncation=truncation,
                        presence_penalty=presence_penalty,
                        extra_body=extra_body,
                        context_management=context_management,
                    ),
                )
            )
        except asyncio.QueueFull:
            raise ValueError(
                f"Background response queue is full (max {BACKGROUND_QUEUE_MAX_SIZE}). Try again later."
            ) from None

        return queued_response

    async def _run_background_response_loop(
        self,
        response_id: str,
        input: str | list[OpenAIResponseInput],
        model: str,
        prompt: OpenAIResponsePrompt | None = None,
        instructions: str | None = None,
        previous_response_id: str | None = None,
        conversation: str | None = None,
        store: bool | None = True,
        temperature: float | None = None,
        frequency_penalty: float | None = None,
        text: OpenAIResponseText | None = None,
        tool_choice: OpenAIResponseInputToolChoice | None = None,
        tools: list[OpenAIResponseInputTool] | None = None,
        include: list[ResponseItemInclude] | None = None,
        max_infer_iters: int | None = 10,
        guardrail_ids: list[str] | None = None,
        parallel_tool_calls: bool | None = None,
        max_tool_calls: int | None = None,
        reasoning: OpenAIResponseReasoning | None = None,
        max_output_tokens: int | None = None,
        safety_identifier: str | None = None,
        service_tier: ServiceTier | None = None,
        metadata: dict[str, str] | None = None,
        truncation: ResponseTruncation | None = None,
        presence_penalty: float | None = None,
        extra_body: dict | None = None,
        context_management: list | None = None,
    ) -> None:
        """Inner loop for background response processing, separated for timeout wrapping."""
        # Check if response was cancelled before starting
        existing = await self.responses_store.get_response_object(response_id)
        if existing.status == "cancelled":
            logger.info("Background response was cancelled before processing started", response_id=response_id)
            return

        # Update status to in_progress
        existing.status = "in_progress"
        await self.responses_store.update_response_object(existing)

        # Process the response using existing streaming logic
        stream_gen = self._create_streaming_response(
            input=input,
            conversation=conversation,
            model=model,
            prompt=prompt,
            instructions=instructions,
            previous_response_id=previous_response_id,
            store=store,
            temperature=temperature,
            frequency_penalty=frequency_penalty,
            text=text,
            tools=tools,
            tool_choice=tool_choice,
            max_infer_iters=max_infer_iters,
            guardrail_ids=guardrail_ids,
            parallel_tool_calls=parallel_tool_calls,
            max_tool_calls=max_tool_calls,
            reasoning=reasoning,
            max_output_tokens=max_output_tokens,
            safety_identifier=safety_identifier,
            service_tier=service_tier,
            metadata=metadata,
            include=include,
            truncation=truncation,
            response_id=response_id,
            presence_penalty=presence_penalty,
            extra_body=extra_body,
            context_management=context_management,
        )

        result_response = None

        async for stream_chunk in stream_gen:
            # Check for cancellation periodically
            current = await self.responses_store.get_response_object(response_id)
            if current.status == "cancelled":
                logger.info("Background response was cancelled during processing", response_id=response_id)
                return

            match stream_chunk.type:
                case "response.completed" | "response.incomplete" | "response.failed":
                    result_response = stream_chunk.response
                case _:
                    pass

        if result_response is not None:
            # Check if response was cancelled before final update to avoid race condition
            current = await self.responses_store.get_response_object(response_id)
            if current.status == "cancelled":
                logger.info("Background response was cancelled before final update", response_id=response_id)
                return

            result_response.background = True
            result_response.id = response_id  # Ensure we update the correct response
            await self.responses_store.update_response_object(result_response)
        else:
            # Something went wrong - mark as failed
            existing = await self.responses_store.get_response_object(response_id)
            if existing.status == "cancelled":
                logger.info("Background response was cancelled before failure update", response_id=response_id)
                return

            existing.status = "failed"
            existing.error = OpenAIResponseError(
                code="processing_error",
                message="Response stream never reached a terminal state",
            )
            await self.responses_store.update_response_object(existing)

    async def _create_streaming_response(
        self,
        input: str | list[OpenAIResponseInput],
        model: str,
        instructions: str | None = None,
        previous_response_id: str | None = None,
        prompt_cache_key: str | None = None,
        conversation: str | None = None,
        prompt: OpenAIResponsePrompt | None = None,
        store: bool | None = True,
        temperature: float | None = None,
        top_p: float | None = None,
        frequency_penalty: float | None = None,
        text: OpenAIResponseText | None = None,
        tools: list[OpenAIResponseInputTool] | None = None,
        tool_choice: OpenAIResponseInputToolChoice | None = None,
        max_infer_iters: int | None = 10,
        guardrail_ids: list[str] | None = None,
        parallel_tool_calls: bool | None = True,
        max_tool_calls: int | None = None,
        reasoning: OpenAIResponseReasoning | None = None,
        max_output_tokens: int | None = None,
        safety_identifier: str | None = None,
        service_tier: ServiceTier | None = None,
        metadata: dict[str, str] | None = None,
        include: list[ResponseItemInclude] | None = None,
        truncation: ResponseTruncation | None = None,
        response_id: str | None = None,
        top_logprobs: int | None = None,
        presence_penalty: float | None = None,
        extra_body: dict | None = None,
        stream_options: ResponseStreamOptions | None = None,
        context_management: list | None = None,
    ) -> AsyncIterator[OpenAIResponseObjectStream]:
        # These should never be None when called from create_openai_response (which sets defaults)
        # but we assert here to help mypy understand the types
        assert text is not None, "text must not be None"
        assert max_infer_iters is not None, "max_infer_iters must not be None"

        # Input preprocessing
        all_input, messages, tool_context, previous_usage = await self._process_input_with_previous_response(
            input, tools, previous_response_id, conversation
        )

        # Auto-compact if context_management is configured (runs on resolved history, not just new input)
        if context_management:
            compacted_input = await self._maybe_auto_compact(all_input, model, context_management, previous_usage)
            if compacted_input is not all_input:
                all_input = compacted_input
                messages = await convert_response_input_to_chat_messages(all_input, files_api=self.files_api)

        if instructions:
            messages.insert(0, OpenAISystemMessageParam(content=instructions))

        # Prepend reusable prompt (if provided)
        await self._prepend_prompt(messages, prompt)

        # Structured outputs
        response_format = await convert_response_text_to_chat_response_format(text)

        ctx = ChatCompletionContext(
            model=model,
            messages=messages,
            response_tools=tools,
            tool_choice=tool_choice,
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            response_format=response_format,
            tool_context=tool_context,
            inputs=all_input,
            extra_body=extra_body,
        )

        # Create orchestrator and delegate streaming logic
        if response_id is None:
            response_id = f"resp_{uuid.uuid4()}"
        created_at = int(time.time())

        # Create a per-request MCP session manager for session reuse (fix for #4452)
        # This avoids redundant tools/list calls when making multiple MCP tool invocations
        async with MCPSessionManager() as mcp_session_manager:
            request_tool_executor = ToolExecutor(
                tool_groups_api=self.tool_groups_api,
                tool_runtime_api=self.tool_runtime_api,
                vector_io_api=self.vector_io_api,
                vector_stores_config=self.tool_executor.vector_stores_config,
                mcp_session_manager=mcp_session_manager,
            )

            orchestrator = StreamingResponseOrchestrator(
                inference_api=self.inference_api,
                ctx=ctx,
                response_id=response_id,
                created_at=created_at,
                prompt=prompt,
                prompt_cache_key=prompt_cache_key,
                text=text,
                max_infer_iters=max_infer_iters,
                parallel_tool_calls=parallel_tool_calls,
                tool_executor=request_tool_executor,
                safety_api=self.safety_api,
                connectors_api=self.connectors_api,
                guardrail_ids=guardrail_ids,
                instructions=instructions,
                max_tool_calls=max_tool_calls,
                reasoning=reasoning,
                max_output_tokens=max_output_tokens,
                safety_identifier=safety_identifier,
                service_tier=service_tier,
                metadata=metadata,
                include=include,
                store=store,
                truncation=truncation,
                top_logprobs=top_logprobs,
                presence_penalty=presence_penalty,
                extra_body=extra_body,
                stream_options=stream_options,
            )

            final_response = None
            failed_response = None

            output_items: list[ConversationItem] = []

            input_items_for_storage = self._prepare_input_items_for_storage(all_input)

            async for stream_chunk in orchestrator.create_response():
                match stream_chunk.type:
                    case "response.completed" | "response.incomplete":
                        final_response = stream_chunk.response
                    case "response.failed":
                        failed_response = stream_chunk.response
                    case "response.output_item.done":
                        item = stream_chunk.item
                        output_items.append(item)
                    case _:
                        pass  # Other event types

                # Incremental persistence: persist on significant state changes
                # This enables clients to poll GET /v1/responses/{response_id} during streaming
                if store:
                    await self._persist_streaming_state(
                        stream_chunk=stream_chunk,
                        orchestrator=orchestrator,
                        input_items=input_items_for_storage,
                        output_items=output_items,
                    )

                # Store and sync before yielding terminal events
                # This ensures the storage/syncing happens even if the consumer breaks after receiving the event
                if (
                    stream_chunk.type in {"response.completed", "response.incomplete"}
                    and final_response
                    and failed_response is None
                    and store
                ):
                    if conversation:
                        messages_to_store = list(
                            filter(lambda x: not isinstance(x, OpenAISystemMessageParam), orchestrator.final_messages)
                        )
                        await self._sync_response_to_conversation(conversation, input, output_items)
                        await self.responses_store.store_conversation_messages(conversation, messages_to_store)

                yield stream_chunk

    async def delete_openai_response(self, response_id: str) -> OpenAIDeleteResponseObject:
        return await self.responses_store.delete_response_object(response_id)

    async def compact_openai_response(
        self,
        model: str,
        input: str | list[OpenAIResponseInput] | None = None,
        instructions: str | None = None,
        previous_response_id: str | None = None,
        prompt_cache_key: str | None = None,
    ) -> OpenAICompactedResponse:
        # Resolve input from previous_response_id or direct input
        resolved_messages = None
        if previous_response_id:
            previous_response = await self.responses_store.get_response_object(previous_response_id)
            if previous_response.status in ("queued", "in_progress"):
                raise ValueError(
                    f"Response {previous_response_id} is still {previous_response.status}. "
                    "Cannot compact an incomplete background response."
                )
            if input is not None:
                all_input = await self._prepend_previous_response(input, previous_response)
            else:
                all_input = list(previous_response.input) + list(previous_response.output)

            # Use stored messages for full conversation context when the caller also
            # provides new input. In conversation= mode, .input only contains the last
            # turn's items while .messages has the complete chat history. When input is
            # None, .input + .output (set above) already captures the full context.
            if previous_response.messages and input is not None:
                message_adapter = TypeAdapter(list[OpenAIMessageParam])
                resolved_messages = message_adapter.validate_python(previous_response.messages)
        elif input is not None:
            if isinstance(input, str):
                all_input = [OpenAIResponseMessage(content=input, role="user")]
            else:
                all_input = list(input)
        else:
            raise InvalidParameterError(
                "input, previous_response_id", None, "Either 'input' or 'previous_response_id' must be provided."
            )

        # Convert to chat messages for the summarization call
        if resolved_messages is not None:
            messages = resolved_messages
            # If caller also provided new input, convert and append it so the
            # summarization covers the full conversation including the new turn.
            if input is not None:
                new_messages = await convert_response_input_to_chat_messages(
                    input, previous_messages=messages, files_api=self.files_api
                )
                messages.extend(new_messages)
        else:
            messages = await convert_response_input_to_chat_messages(all_input, files_api=self.files_api)

        # Add summarization prompt (use config template, prepend instructions if provided)
        summarization_prompt = self.compaction_config.summarization_prompt
        if instructions:
            summarization_prompt = f"{instructions}\n\n{summarization_prompt}"

        messages.append(OpenAIUserMessageParam(role="user", content=summarization_prompt))

        # Call inference to generate the summary (use configured model or fall back to conversation model)
        summarization_model = self.compaction_config.summarization_model or model
        params = OpenAIChatCompletionRequestWithExtraBody(
            model=summarization_model,
            messages=messages,
            stream=False,
            prompt_cache_key=prompt_cache_key,
        )
        completion = await self.inference_api.openai_chat_completion(params)
        assert not isinstance(completion, AsyncIterator)

        # Extract summary text from the completion
        summary_text = ""
        if hasattr(completion, "choices") and completion.choices:
            choice = completion.choices[0]
            if choice.message and choice.message.content:
                summary_text = choice.message.content

        # Extract user messages from input (matching OpenAI behavior: all user messages verbatim)
        output_items: list[OpenAIResponseInput] = []
        for item in all_input:
            if isinstance(item, OpenAIResponseMessage) and item.role == "user":
                output_items.append(
                    OpenAIResponseMessage(
                        id=f"msg_{uuid.uuid4().hex[:24]}",
                        type="message",
                        status="completed",
                        role="user",
                        content=item.content,
                    )
                )

        # Prepend the summary prefix to frame the summary as a handoff
        if self.compaction_config.summary_prefix:
            summary_text = f"{self.compaction_config.summary_prefix}\n{summary_text}"

        # Add compaction item as last element
        compaction_item = OpenAIResponseCompaction(
            id=f"cmp_{uuid.uuid4().hex[:24]}",
            encrypted_content=summary_text,
        )
        output_items.append(compaction_item)

        # Build usage from completion
        usage = completion.usage
        usage_data = OpenAIResponseUsage(
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            total_tokens=usage.total_tokens if usage else 0,
            input_tokens_details=OpenAIResponseUsageInputTokensDetails(
                cached_tokens=usage.prompt_tokens_details.cached_tokens
                if usage and usage.prompt_tokens_details and usage.prompt_tokens_details.cached_tokens is not None
                else 0
            ),
            output_tokens_details=OpenAIResponseUsageOutputTokensDetails(
                reasoning_tokens=usage.completion_tokens_details.reasoning_tokens
                if usage
                and usage.completion_tokens_details
                and usage.completion_tokens_details.reasoning_tokens is not None
                else 0
            ),
        )

        response_id = f"resp_{uuid.uuid4().hex[:24]}"
        created_at = int(time.time())

        # Store a full OpenAIResponseObject so the compacted response ID is
        # retrievable and can be used as previous_response_id in subsequent calls.
        stored_response = OpenAIResponseObject(
            id=response_id,
            created_at=created_at,
            model=model,
            status="completed",
            output=[],
            usage=usage_data,
            store=True,
        )

        # Convert compacted output to chat messages for conversation continuity
        compacted_messages = await convert_response_input_to_chat_messages(output_items, files_api=self.files_api)

        await self.responses_store.store_response_object(
            response_object=stored_response,
            input=output_items,
            messages=compacted_messages,
        )

        return OpenAICompactedResponse(
            id=response_id,
            created_at=created_at,
            output=output_items,
            usage=usage_data,
        )

    def _count_tokens(self, input: str | list[OpenAIResponseInput], model: str = "") -> int:
        """Estimate token count using tiktoken. Used as fallback when provider usage is unavailable."""
        # Use explicitly configured encoding, or resolve from model name
        if self.compaction_config.tokenizer_encoding:
            encoding = tiktoken.get_encoding(self.compaction_config.tokenizer_encoding)
        else:
            # Strip provider prefix (e.g. "openai/gpt-4o" -> "gpt-4o") for tiktoken lookup
            model_name = model.split("/")[-1] if "/" in model else model
            try:
                encoding = tiktoken.encoding_for_model(model_name)
            except KeyError as e:
                raise ValueError(
                    f"Failed to resolve tiktoken encoding for model '{model}'. "
                    "Set 'tokenizer_encoding' in compaction_config (e.g. 'o200k_base') "
                    "or use an OpenAI model name that tiktoken recognizes."
                ) from e

        if isinstance(input, str):
            return len(encoding.encode(input))

        total_tokens = 0
        for item in input:
            if isinstance(item, OpenAIResponseMessage):
                if isinstance(item.content, str):
                    total_tokens += len(encoding.encode(item.content))
                elif isinstance(item.content, list):
                    for part in item.content:
                        if hasattr(part, "text"):
                            total_tokens += len(encoding.encode(part.text))
            elif isinstance(item, OpenAIResponseCompaction):
                total_tokens += len(encoding.encode(item.encrypted_content))
            elif hasattr(item, "arguments"):
                args = getattr(item, "arguments", "")
                if args:
                    total_tokens += len(encoding.encode(args))
            elif hasattr(item, "output"):
                output = getattr(item, "output", "")
                if isinstance(output, str):
                    total_tokens += len(encoding.encode(output))
        return total_tokens

    async def _maybe_auto_compact(
        self,
        input: str | list[OpenAIResponseInput],
        model: str,
        context_management: list,
        previous_usage: OpenAIResponseUsage | None = None,
    ) -> str | list[OpenAIResponseInput]:
        """Auto-compact input if token count exceeds compact_threshold."""
        for entry in context_management:
            entry_type = entry.type if hasattr(entry, "type") else entry.get("type")
            if entry_type != "compaction":
                continue

            threshold = (
                entry.compact_threshold if hasattr(entry, "compact_threshold") else entry.get("compact_threshold")
            )
            if threshold is None:
                threshold = self.compaction_config.default_compact_threshold
            if threshold is None:
                continue

            # Use provider-reported token count when available, fall back to tiktoken estimate
            if previous_usage and previous_usage.total_tokens:
                token_count = previous_usage.total_tokens
            else:
                token_count = self._count_tokens(input, model=model)
            if token_count > threshold:
                logger.debug("Auto-compacting", token_count=token_count, threshold=threshold)
                compacted = await self.compact_openai_response(model=model, input=input)
                return list(compacted.output)

        return input

    async def cancel_openai_response(
        self,
        response_id: str,
    ) -> OpenAIResponseObject:
        """Cancel a response that is queued or in progress.

        Args:
            response_id: The ID of the response to cancel

        Returns:
            The updated response object with status "cancelled"

        Raises:
            ResponseNotFoundError: If the response doesn't exist (automatically from store)
            ConflictError: If the response is already in a terminal state
        """
        # Get current response state
        response = await self.responses_store.get_response_object(response_id)

        # If already cancelled, return current state (idempotent)
        if response.status == "cancelled":
            return response.to_response_object()

        # Only background responses can be cancelled
        if not response.background:
            raise ConflictError(f"Cannot cancel response '{response_id}': only background responses can be cancelled")

        # Cannot cancel responses in terminal states
        if response.status in ["completed", "failed", "incomplete"]:
            raise ConflictError(f"Cannot cancel response '{response_id}' with status '{response.status}'")

        # Update status to cancelled in database
        response.status = "cancelled"
        await self.responses_store.update_response_object(response)

        # If the response is currently being processed, cancel the task
        async with self._background_response_tasks_lock:
            task = self._background_response_tasks.get(response_id)
            if task:
                task.cancel()
                # Note: task removal handled in worker's finally block

        # Re-fetch from store to return the persisted state
        updated = await self.responses_store.get_response_object(response_id)
        return updated.to_response_object()

    async def _sync_response_to_conversation(
        self, conversation_id: str, input: str | list[OpenAIResponseInput] | None, output_items: list[ConversationItem]
    ) -> None:
        """Sync content and response messages to the conversation."""
        # Type as ConversationItem union to avoid list invariance issues
        conversation_items: list[ConversationItem] = []

        if isinstance(input, str):
            conversation_items.append(
                OpenAIResponseMessage(role="user", content=[OpenAIResponseInputMessageContentText(text=input)])
            )
        elif isinstance(input, list):
            conversation_items.extend(item for item in input if not isinstance(item, OpenAIResponseCompaction))

        conversation_items.extend(output_items)

        adapter = TypeAdapter(list[ConversationItem])
        validated_items = adapter.validate_python(conversation_items)
        await self.conversations_api.add_items(conversation_id, AddItemsRequest(items=validated_items))
