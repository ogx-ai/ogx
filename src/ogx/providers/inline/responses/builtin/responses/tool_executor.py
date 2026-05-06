# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any

from opentelemetry import trace

from ogx.core.datatypes import VectorStoresConfig
from ogx.log import get_logger
from ogx_api import (
    ImageContentItem,
    OpenAIChatCompletionContentPartImageParam,
    OpenAIChatCompletionContentPartTextParam,
    OpenAIChatCompletionToolCall,
    OpenAIImageURL,
    OpenAIResponseInputToolFileSearch,
    OpenAIResponseInputToolMCP,
    OpenAIResponseObjectStreamResponseFileSearchCallCompleted,
    OpenAIResponseObjectStreamResponseFileSearchCallInProgress,
    OpenAIResponseObjectStreamResponseFileSearchCallSearching,
    OpenAIResponseObjectStreamResponseMcpCallCompleted,
    OpenAIResponseObjectStreamResponseMcpCallFailed,
    OpenAIResponseObjectStreamResponseMcpCallInProgress,
    OpenAIResponseObjectStreamResponseWebSearchCallCompleted,
    OpenAIResponseObjectStreamResponseWebSearchCallInProgress,
    OpenAIResponseObjectStreamResponseWebSearchCallSearching,
    OpenAIResponseOutputMessageFileSearchToolCall,
    OpenAIResponseOutputMessageFileSearchToolCallResults,
    OpenAIResponseOutputMessageWebSearchToolCall,
    OpenAISearchVectorStoreRequest,
    OpenAIToolMessageParam,
    TextContentItem,
    ToolGroups,
    ToolInvocationResult,
    ToolRuntime,
    VectorIO,
)

from .types import ChatCompletionContext, ToolExecutionResult

logger = get_logger(name=__name__, category="agents::builtin")
tracer = trace.get_tracer(__name__)


class ToolExecutor:
    """Executes tool calls including file search, web search, MCP, and function tools."""

    def __init__(
        self,
        tool_groups_api: ToolGroups,
        tool_runtime_api: ToolRuntime,
        vector_io_api: VectorIO,
        vector_stores_config=None,
        mcp_session_manager=None,
    ):
        self.tool_groups_api = tool_groups_api
        self.tool_runtime_api = tool_runtime_api
        self.vector_io_api = vector_io_api
        self.vector_stores_config = vector_stores_config
        # Optional MCPSessionManager for session reuse within a request (fix for #4452)
        self.mcp_session_manager = mcp_session_manager

    async def execute_tool_call(
        self,
        tool_call: OpenAIChatCompletionToolCall,
        ctx: ChatCompletionContext,
        sequence_number: int,
        output_index: int,
        item_id: str,
        mcp_tool_to_server: dict[str, OpenAIResponseInputToolMCP] | None = None,
    ) -> AsyncIterator[ToolExecutionResult]:
        tool_call_id = tool_call.id
        function = tool_call.function
        tool_kwargs = json.loads(function.arguments) if function and function.arguments else {}

        if not function or not tool_call_id or not function.name:
            yield ToolExecutionResult(sequence_number=sequence_number)
            return

        # Emit progress events for tool execution start
        async for event_result in self._emit_progress_events(
            function.name, ctx, sequence_number, output_index, item_id, mcp_tool_to_server
        ):
            sequence_number = event_result.sequence_number
            yield event_result

        # Execute the actual tool call
        error_exc, result = await self._execute_tool(function.name, tool_kwargs, ctx, mcp_tool_to_server)

        # Emit completion events for tool execution
        has_error = bool(
            error_exc
            or (
                result
                and (
                    ((error_code := getattr(result, "error_code", None)) and error_code > 0)
                    or getattr(result, "error_message", None)
                )
            )
        )
        async for event_result in self._emit_completion_events(
            function.name, ctx, sequence_number, output_index, item_id, has_error, mcp_tool_to_server
        ):
            sequence_number = event_result.sequence_number
            yield event_result

        # Build result messages from tool execution
        output_message, input_message = await self._build_result_messages(
            function, tool_call_id, item_id, tool_kwargs, ctx, error_exc, result, has_error, mcp_tool_to_server
        )

        # Yield the final result
        yield ToolExecutionResult(
            sequence_number=sequence_number,
            final_output_message=output_message,
            final_input_message=input_message,
            citation_files=(
                metadata.get("citation_files") if result and (metadata := getattr(result, "metadata", None)) else None
            ),
        )

    async def _execute_file_search_via_vector_store(
        self,
        query: str,
        response_file_search_tool: OpenAIResponseInputToolFileSearch,
    ) -> ToolInvocationResult:
        """Execute file search using vector_stores.search API with filters support."""
        search_results = []

        # Create search tasks for all vector stores
        async def search_single_store(vector_store_id):
            try:
                # Use default_search_mode from config if available
                search_mode = "vector"
                if (
                    self.vector_stores_config
                    and hasattr(self.vector_stores_config, "chunk_retrieval_params")
                    and hasattr(self.vector_stores_config.chunk_retrieval_params, "default_search_mode")
                ):
                    search_mode = self.vector_stores_config.chunk_retrieval_params.default_search_mode

                search_response = await self.vector_io_api.openai_search_vector_store(
                    vector_store_id=vector_store_id,
                    request=OpenAISearchVectorStoreRequest(
                        query=query,
                        filters=response_file_search_tool.filters,
                        max_num_results=response_file_search_tool.max_num_results,
                        ranking_options=response_file_search_tool.ranking_options,
                        rewrite_query=False,
                        search_mode=search_mode,
                    ),
                )
                return search_response.data
            except Exception as e:
                logger.warning("Failed to search vector store", vector_store_id=vector_store_id, error=str(e))
                return []

        # Run all searches in parallel using gather
        search_tasks = [search_single_store(vid) for vid in response_file_search_tool.vector_store_ids]
        all_results = await asyncio.gather(*search_tasks)

        # Flatten results
        for results in all_results:
            search_results.extend(results)

        # Get templates from vector stores config, fallback to constants

        # Check if annotations are enabled
        enable_annotations = (
            self.vector_stores_config
            and self.vector_stores_config.annotation_prompt_params
            and self.vector_stores_config.annotation_prompt_params.enable_annotations
        )

        # Get templates
        header_template = self.vector_stores_config.file_search_params.header_template
        footer_template = self.vector_stores_config.file_search_params.footer_template
        context_template = self.vector_stores_config.context_prompt_params.context_template

        # Get annotation templates (use defaults if annotations disabled)
        if enable_annotations:
            chunk_annotation_template = self.vector_stores_config.annotation_prompt_params.chunk_annotation_template
            annotation_instruction_template = (
                self.vector_stores_config.annotation_prompt_params.annotation_instruction_template
            )
        else:
            # Use defaults from VectorStoresConfig when annotations disabled
            default_config = VectorStoresConfig()
            chunk_annotation_template = default_config.annotation_prompt_params.chunk_annotation_template
            annotation_instruction_template = default_config.annotation_prompt_params.annotation_instruction_template

        content_items = []
        content_items.append(TextContentItem(text=header_template.format(num_chunks=len(search_results))))

        unique_files = set()
        for i, result_item in enumerate(search_results):
            chunk_text = result_item.content[0].text if result_item.content else ""
            # Get file_id from attributes if result_item.file_id is empty
            file_id = result_item.file_id or (
                result_item.attributes.get("document_id") if result_item.attributes else None
            )
            metadata_text = f"document_id: {file_id}, score: {result_item.score}"
            if result_item.attributes:
                metadata_text += f", attributes: {result_item.attributes}"

            text_content = chunk_annotation_template.format(
                index=i + 1, metadata_text=metadata_text, file_id=file_id, chunk_text=chunk_text
            )
            content_items.append(TextContentItem(text=text_content))
            unique_files.add(file_id)

        content_items.append(TextContentItem(text=footer_template))

        annotation_instruction = ""
        if unique_files:
            annotation_instruction = annotation_instruction_template

        content_items.append(
            TextContentItem(
                text=context_template.format(
                    query=query, num_chunks=len(search_results), annotation_instruction=annotation_instruction
                )
            )
        )

        # handling missing attributes for old versions
        citation_files = {}
        for result in search_results:
            file_id = result.file_id
            if not file_id and result.attributes:
                file_id = result.attributes.get("document_id")

            filename = result.filename
            if not filename and result.attributes:
                filename = result.attributes.get("filename")
            if not filename:
                filename = "unknown"

            citation_files[file_id] = filename

        # Cast to proper InterleavedContent type (list invariance)
        return ToolInvocationResult(
            content=content_items,  # type: ignore[arg-type]
            metadata={
                "document_ids": [r.file_id for r in search_results],
                "chunks": [r.content[0].text if r.content else "" for r in search_results],
                "scores": [r.score for r in search_results],
                "attributes": [r.attributes or {} for r in search_results],
                "citation_files": citation_files,
            },
        )

    async def _emit_progress_events(
        self,
        function_name: str,
        ctx: ChatCompletionContext,
        sequence_number: int,
        output_index: int,
        item_id: str,
        mcp_tool_to_server: dict[str, OpenAIResponseInputToolMCP] | None = None,
    ) -> AsyncIterator[ToolExecutionResult]:
        """Emit progress events for tool execution start."""
        # Emit in_progress event based on tool type (only for tools with specific streaming events)
        if mcp_tool_to_server and function_name in mcp_tool_to_server:
            sequence_number += 1
            yield ToolExecutionResult(
                stream_event=OpenAIResponseObjectStreamResponseMcpCallInProgress(
                    item_id=item_id,
                    output_index=output_index,
                    sequence_number=sequence_number,
                ),
                sequence_number=sequence_number,
            )
        elif function_name == "web_search":
            sequence_number += 1
            yield ToolExecutionResult(
                stream_event=OpenAIResponseObjectStreamResponseWebSearchCallInProgress(
                    item_id=item_id,
                    output_index=output_index,
                    sequence_number=sequence_number,
                ),
                sequence_number=sequence_number,
            )
        elif function_name in ("knowledge_search", "file_search"):
            sequence_number += 1
            yield ToolExecutionResult(
                stream_event=OpenAIResponseObjectStreamResponseFileSearchCallInProgress(
                    item_id=item_id,
                    output_index=output_index,
                    sequence_number=sequence_number,
                ),
                sequence_number=sequence_number,
            )

        # For web search, emit searching event
        if function_name == "web_search":
            sequence_number += 1
            yield ToolExecutionResult(
                stream_event=OpenAIResponseObjectStreamResponseWebSearchCallSearching(
                    item_id=item_id,
                    output_index=output_index,
                    sequence_number=sequence_number,
                ),
                sequence_number=sequence_number,
            )

        # For file search, emit searching event
        if function_name in ("knowledge_search", "file_search"):
            sequence_number += 1
            yield ToolExecutionResult(
                stream_event=OpenAIResponseObjectStreamResponseFileSearchCallSearching(
                    item_id=item_id,
                    output_index=output_index,
                    sequence_number=sequence_number,
                ),
                sequence_number=sequence_number,
            )

    async def _execute_tool(
        self,
        function_name: str,
        tool_kwargs: dict,
        ctx: ChatCompletionContext,
        mcp_tool_to_server: dict[str, OpenAIResponseInputToolMCP] | None = None,
    ) -> tuple[Exception | None, Any]:
        """Execute the tool and return error exception and result."""
        error_exc = None
        result = None

        try:
            if mcp_tool_to_server and function_name in mcp_tool_to_server:
                from ogx.providers.utils.tools.mcp import invoke_mcp_tool

                mcp_tool = mcp_tool_to_server[function_name]
                if not mcp_tool.server_url:
                    raise ValueError(f"Failed to invoke MCP tool {function_name}: server_url is not set")
                attributes = {
                    "server_label": mcp_tool.server_label,
                    "server_url": mcp_tool.server_url,
                    "tool_name": function_name,
                }
                # TODO: follow semantic conventions for Open Telemetry tool spans
                # https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-spans/#execute-tool-span
                with tracer.start_as_current_span("invoke_mcp_tool", attributes=attributes):
                    # Pass session_manager for session reuse within request (fix for #4452)
                    result = await invoke_mcp_tool(
                        endpoint=mcp_tool.server_url,
                        tool_name=function_name,
                        kwargs=tool_kwargs,
                        headers=mcp_tool.headers,
                        authorization=mcp_tool.authorization,
                        session_manager=self.mcp_session_manager,
                    )
            elif function_name in ("knowledge_search", "file_search"):
                response_file_search_tool = (
                    next(
                        (t for t in ctx.response_tools if isinstance(t, OpenAIResponseInputToolFileSearch)),
                        None,
                    )
                    if ctx.response_tools
                    else None
                )
                if response_file_search_tool:
                    # Use vector_stores.search API instead of file_search tool
                    # to support filters and ranking_options
                    query = tool_kwargs.get("query", "")
                    with tracer.start_as_current_span(function_name):
                        result = await self._execute_file_search_via_vector_store(
                            query=query,
                            response_file_search_tool=response_file_search_tool,
                        )
            else:
                attributes = {
                    "tool_name": function_name,
                }
                # TODO: follow semantic conventions for Open Telemetry tool spans
                # https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-spans/#execute-tool-span
                with tracer.start_as_current_span("invoke_tool", attributes=attributes):
                    result = await self.tool_runtime_api.invoke_tool(
                        tool_name=function_name,
                        kwargs=tool_kwargs,
                    )
        except Exception as e:
            error_exc = e

        return error_exc, result

    async def _emit_completion_events(
        self,
        function_name: str,
        ctx: ChatCompletionContext,
        sequence_number: int,
        output_index: int,
        item_id: str,
        has_error: bool,
        mcp_tool_to_server: dict[str, OpenAIResponseInputToolMCP] | None = None,
    ) -> AsyncIterator[ToolExecutionResult]:
        """Emit completion or failure events for tool execution."""
        if mcp_tool_to_server and function_name in mcp_tool_to_server:
            sequence_number += 1
            if has_error:
                mcp_failed_event = OpenAIResponseObjectStreamResponseMcpCallFailed(
                    sequence_number=sequence_number,
                )
                yield ToolExecutionResult(stream_event=mcp_failed_event, sequence_number=sequence_number)
            else:
                mcp_completed_event = OpenAIResponseObjectStreamResponseMcpCallCompleted(
                    sequence_number=sequence_number,
                )
                yield ToolExecutionResult(stream_event=mcp_completed_event, sequence_number=sequence_number)
        elif function_name == "web_search":
            sequence_number += 1
            web_completion_event = OpenAIResponseObjectStreamResponseWebSearchCallCompleted(
                item_id=item_id,
                output_index=output_index,
                sequence_number=sequence_number,
            )
            yield ToolExecutionResult(stream_event=web_completion_event, sequence_number=sequence_number)
        elif function_name in ("knowledge_search", "file_search"):
            sequence_number += 1
            file_completion_event = OpenAIResponseObjectStreamResponseFileSearchCallCompleted(
                item_id=item_id,
                output_index=output_index,
                sequence_number=sequence_number,
            )
            yield ToolExecutionResult(stream_event=file_completion_event, sequence_number=sequence_number)

    async def _build_result_messages(
        self,
        function,
        tool_call_id: str,
        item_id: str,
        tool_kwargs: dict,
        ctx: ChatCompletionContext,
        error_exc: Exception | None,
        result: Any,
        has_error: bool,
        mcp_tool_to_server: dict[str, OpenAIResponseInputToolMCP] | None = None,
    ) -> tuple[Any, Any]:
        """Build output and input messages from tool execution results."""
        from ogx.providers.utils.inference.prompt_adapter import (
            interleaved_content_as_str,
        )

        # Build output message
        message: Any
        if mcp_tool_to_server and function.name in mcp_tool_to_server:
            from ogx_api import (
                OpenAIResponseOutputMessageMCPCall,
            )

            message = OpenAIResponseOutputMessageMCPCall(
                id=item_id,
                arguments=function.arguments,
                name=function.name,
                server_label=mcp_tool_to_server[function.name].server_label,
            )
            if error_exc:
                message.error = str(error_exc)
            elif (result and (error_code := getattr(result, "error_code", None)) and error_code > 0) or (
                result and getattr(result, "error_message", None)
            ):
                ec = getattr(result, "error_code", "unknown")
                em = getattr(result, "error_message", "")
                message.error = f"Error (code {ec}): {em}"
            elif result and (content := getattr(result, "content", None)):
                message.output = interleaved_content_as_str(content)
        else:
            if function.name == "web_search":
                message = OpenAIResponseOutputMessageWebSearchToolCall(
                    id=item_id,
                    status="completed",
                )
                if has_error:
                    message.status = "failed"
            elif function.name in ("knowledge_search", "file_search"):
                message = OpenAIResponseOutputMessageFileSearchToolCall(
                    id=item_id,
                    queries=[tool_kwargs.get("query", "")],
                    status="completed",
                )
                if result and (metadata := getattr(result, "metadata", None)) and "document_ids" in metadata:
                    message.results = []
                    attributes_list = metadata.get("attributes", [])
                    for i, doc_id in enumerate(metadata["document_ids"]):
                        text = metadata["chunks"][i] if "chunks" in metadata else None
                        score = metadata["scores"][i] if "scores" in metadata else None
                        attrs = attributes_list[i] if i < len(attributes_list) else {}
                        message.results.append(
                            OpenAIResponseOutputMessageFileSearchToolCallResults(
                                file_id=doc_id,
                                filename=doc_id,
                                text=text if text is not None else "",
                                score=score if score is not None else 0.0,
                                attributes=attrs,
                            )
                        )
                if has_error:
                    message.status = "failed"
            else:
                raise ValueError(f"Unknown tool {function.name} called")

        # Build input message
        input_message: OpenAIToolMessageParam | None = None
        if result and (result_content := getattr(result, "content", None)):
            # all the mypy contortions here are still unsatisfactory with random Any typing
            if isinstance(result_content, str):
                msg_content: str | list[Any] = result_content
            elif isinstance(result_content, list):
                content_list: list[Any] = []
                for item in result_content:
                    part: Any
                    if isinstance(item, TextContentItem):
                        part = OpenAIChatCompletionContentPartTextParam(text=item.text)
                    elif isinstance(item, ImageContentItem):
                        if item.image.data:
                            url_value = f"data:image;base64,{item.image.data}"
                        else:
                            url_value = str(item.image.url) if item.image.url else ""
                        part = OpenAIChatCompletionContentPartImageParam(image_url=OpenAIImageURL(url=url_value))
                    else:
                        raise ValueError(f"Unknown result content type: {type(item)}")
                    content_list.append(part)
                msg_content = content_list
            else:
                raise ValueError(f"Unknown result content type: {type(result_content)}")
            # OpenAIToolMessageParam accepts str | list[TextParam] but we may have images
            # This is runtime-safe as the API accepts it, but mypy complains
            input_message = OpenAIToolMessageParam(content=msg_content, tool_call_id=tool_call_id)  # type: ignore[arg-type]
        else:
            text = str(error_exc) if error_exc else "Tool execution failed"
            input_message = OpenAIToolMessageParam(content=text, tool_call_id=tool_call_id)

        return message, input_message
