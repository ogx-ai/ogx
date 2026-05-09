# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
import json
import logging
from collections.abc import AsyncIterator, Callable, Iterator
from typing import (
    Any,
    TypedDict,
)
from uuid import uuid4

from ..._types import Headers
from .client_tool import ClientTool, client_tool
from .event_synthesizer import TurnEventSynthesizer
from .tool_parser import ToolParser
from .turn_events import (
    AgentStreamChunk,
    StepCompleted,
    StepProgress,
    StepStarted,
    ToolCallIssuedDelta,
    ToolExecutionStepResult,
    TurnFailed,
)
from .types import CompletionMessage, ToolCall, ToolResponse


class ToolResponsePayload(TypedDict, total=False):
    call_id: str
    tool_name: str
    content: Any
    metadata: dict[str, Any]


logger = logging.getLogger(__name__)


class ToolUtils:
    @staticmethod
    def coerce_tool_content(content: Any) -> str:
        if isinstance(content, str):
            return content
        if content is None:
            return ""
        if isinstance(content, dict | list):
            try:
                return json.dumps(content)
            except TypeError:
                return str(content)
        return str(content)

    @staticmethod
    def parse_tool_arguments(arguments: Any) -> dict[str, Any]:
        if isinstance(arguments, dict):
            return arguments
        if not arguments:
            return {}
        if isinstance(arguments, str):
            try:
                parsed = json.loads(arguments)
            except json.JSONDecodeError:
                logger.warning("Failed to decode tool arguments JSON", exc_info=True)
                return {}
            if isinstance(parsed, dict):
                return parsed
            logger.warning("Tool arguments JSON did not decode into a dict: %s", type(parsed))
            return {}
        logger.warning("Unsupported tool arguments type: %s", type(arguments))
        return {}

    @staticmethod
    def normalize_tool_response(tool_response: Any) -> ToolResponsePayload:
        if isinstance(tool_response, ToolResponse):
            payload: ToolResponsePayload = {
                "call_id": tool_response.call_id,
                "tool_name": str(tool_response.tool_name),
                "content": ToolUtils.coerce_tool_content(tool_response.content),
                "metadata": dict(tool_response.metadata),
            }
            return payload

        if isinstance(tool_response, dict):
            call_id = tool_response.get("call_id")
            tool_name = tool_response.get("tool_name")
            if call_id is None or tool_name is None:
                raise KeyError("Tool response missing required keys 'call_id' or 'tool_name'")
            payload: ToolResponsePayload = {
                "call_id": str(call_id),
                "tool_name": str(tool_name),
                "content": ToolUtils.coerce_tool_content(tool_response.get("content")),
                "metadata": dict(tool_response.get("metadata") or {}),
            }
            return payload

        raise TypeError(f"Unsupported tool response type: {type(tool_response)!r}")


class Agent:
    def __init__(
        self,
        client: Any,  # Accept any OpenAI-compatible client
        *,
        model: str,
        instructions: str,
        tools: list[dict[str, Any] | ClientTool | Callable[..., Any]] | None = None,
        tool_parser: ToolParser | None = None,
        extra_headers: Headers | None = None,
    ):
        """Construct an Agent backed by the responses + conversations APIs.

        Args:
            client: An OpenAI-compatible client (e.g., openai.OpenAI()).
                    The client must support the responses and conversations APIs.
        """
        self.client = client
        self.tool_parser = tool_parser
        self.extra_headers = extra_headers
        self._model = model
        self._instructions = instructions

        # Convert all tools to API format and separate client-side functions
        self._tools, client_tools = AgentUtils.normalize_tools(tools)
        self.client_tools = {tool.get_name(): tool for tool in client_tools}

        self.sessions: list[str] = []

    def create_session(self, session_name: str) -> str:
        conversation = self.client.conversations.create(
            extra_headers=self.extra_headers,
            metadata={"name": session_name},
        )
        self.sessions.append(conversation.id)
        return conversation.id

    def _run_tool_calls(self, tool_calls: list[ToolCall]) -> list[ToolResponsePayload]:
        responses: list[ToolResponsePayload] = []
        for tool_call in tool_calls:
            raw_result = self._run_single_tool(tool_call)
            responses.append(ToolUtils.normalize_tool_response(raw_result))
        return responses

    def _run_single_tool(self, tool_call: ToolCall) -> Any:
        # Execute client-side tools
        if tool_call.tool_name in self.client_tools:
            tool = self.client_tools[tool_call.tool_name]
            result_message = tool.run(
                [
                    CompletionMessage(
                        role="assistant",
                        content=tool_call.arguments,
                        tool_calls=[tool_call],
                        stop_reason="end_of_turn",
                    )
                ]
            )
            return result_message

        # Server-side tools should never reach here (they execute within response stream)
        # If we get here, it's an error
        return {
            "call_id": tool_call.call_id,
            "tool_name": tool_call.tool_name,
            "content": f"Unknown tool `{tool_call.tool_name}` was called.",
        }

    def create_turn(
        self,
        messages: list[dict[str, Any]],
        session_id: str,
        stream: bool = True,
        # TODO: deprecate this
        extra_headers: Headers | None = None,
    ) -> Iterator[AgentStreamChunk] | Any:
        if stream:
            return self._create_turn_streaming(messages, session_id, extra_headers=extra_headers or self.extra_headers)
        else:
            last_chunk: AgentStreamChunk | None = None
            for chunk in self._create_turn_streaming(
                messages, session_id, extra_headers=extra_headers or self.extra_headers
            ):
                last_chunk = chunk

            if not last_chunk or not last_chunk.response:
                raise Exception("Turn did not complete")

            return last_chunk.response

    def _create_turn_streaming(
        self,
        messages: list[dict[str, Any]],
        session_id: str,
        # TODO: deprecate this
        extra_headers: Headers | None = None,
    ) -> Iterator[AgentStreamChunk]:
        # Generate turn_id
        turn_id = f"turn_{uuid4().hex[:12]}"

        # Create synthesizer
        synthesizer = TurnEventSynthesizer(session_id=session_id, turn_id=turn_id)

        request_headers = extra_headers or self.extra_headers

        # Main turn loop
        while True:
            # Create response stream
            raw_stream = self.client.responses.create(
                model=self._model,
                instructions=self._instructions,
                conversation=session_id,
                input=messages,
                tools=self._tools,
                stream=True,
                extra_headers=request_headers,
            )

            # Process events
            function_calls_to_execute: list[ToolCall] = []  # Only client-side!

            for high_level_event in synthesizer.process_raw_stream(raw_stream):
                # Handle failures
                if isinstance(high_level_event, TurnFailed):
                    yield AgentStreamChunk(event=high_level_event)
                    return

                # Track function calls that need client execution
                if isinstance(high_level_event, StepCompleted):
                    if high_level_event.step_type == "inference":
                        result = high_level_event.result
                        if result.function_calls:  # Only client-side function calls
                            function_calls_to_execute = result.function_calls

                yield AgentStreamChunk(event=high_level_event)

            # If no client-side function calls, turn is done
            if not function_calls_to_execute:
                # Emit TurnCompleted
                response = synthesizer.last_response
                if not response:
                    raise RuntimeError("No response available")
                for event in synthesizer.finish_turn():
                    yield AgentStreamChunk(event=event, response=response)
                break

            # Execute client-side tools (emit tool execution step events)
            tool_step_id = f"{turn_id}_step_{synthesizer.step_counter}"
            synthesizer.step_counter += 1

            yield AgentStreamChunk(
                event=StepStarted(
                    step_id=tool_step_id,
                    step_type="tool_execution",
                    turn_id=turn_id,
                    metadata={"server_side": False},
                )
            )

            tool_responses = self._run_tool_calls(function_calls_to_execute)

            yield AgentStreamChunk(
                event=StepCompleted(
                    step_id=tool_step_id,
                    step_type="tool_execution",
                    turn_id=turn_id,
                    result=ToolExecutionStepResult(
                        step_id=tool_step_id,
                        tool_calls=function_calls_to_execute,
                        tool_responses=tool_responses,
                    ),
                )
            )

            # Continue loop with tool outputs as input
            messages = [
                {
                    "type": "function_call_output",
                    "call_id": payload["call_id"],
                    "output": payload["content"],
                }
                for payload in tool_responses
            ]


class AsyncAgent:
    def __init__(
        self,
        client: Any,  # Accept any async OpenAI-compatible client
        *,
        model: str,
        instructions: str,
        tools: list[dict[str, Any] | ClientTool | Callable[..., Any]] | None = None,
        tool_parser: ToolParser | None = None,
        extra_headers: Headers | None = None,
    ):
        """Construct an async Agent backed by the responses + conversations APIs.

        Args:
            client: An async OpenAI-compatible client (e.g., openai.AsyncOpenAI() or AsyncLlamaStackClient).
                    The client must support the responses and conversations APIs.
        """
        self.client = client

        self.tool_parser = tool_parser
        self.extra_headers = extra_headers
        self._model = model
        self._instructions = instructions

        # Convert all tools to API format and separate client-side functions
        self._tools, client_tools = AgentUtils.normalize_tools(tools)
        self.client_tools = {tool.get_name(): tool for tool in client_tools}

        self.sessions: list[str] = []

    async def create_session(self, session_name: str) -> str:
        conversation = await self.client.conversations.create(  # type: ignore[union-attr]
            extra_headers=self.extra_headers,
            metadata={"name": session_name},
        )
        self.sessions.append(conversation.id)
        return conversation.id

    async def create_turn(
        self,
        messages: list[dict[str, Any]],
        session_id: str,
        stream: bool = True,
    ) -> AsyncIterator[AgentStreamChunk] | Any:
        if stream:
            return self._create_turn_streaming(messages, session_id)
        else:
            last_chunk: AgentStreamChunk | None = None
            async for chunk in self._create_turn_streaming(messages, session_id):
                last_chunk = chunk
            if not last_chunk or not last_chunk.response:
                raise Exception("Turn did not complete")
            return last_chunk.response

    async def _run_tool_calls(self, tool_calls: list[ToolCall]) -> list[ToolResponsePayload]:
        responses: list[ToolResponsePayload] = []
        for tool_call in tool_calls:
            raw_result = await self._run_single_tool(tool_call)
            responses.append(ToolUtils.normalize_tool_response(raw_result))
        return responses

    async def _run_single_tool(self, tool_call: ToolCall) -> Any:
        # Execute client-side tools
        if tool_call.tool_name in self.client_tools:
            tool = self.client_tools[tool_call.tool_name]
            result_message = await tool.async_run(
                [
                    CompletionMessage(
                        role="assistant",
                        content=tool_call.arguments,
                        tool_calls=[tool_call],
                        stop_reason="end_of_turn",
                    )
                ]
            )
            return result_message

        # Server-side tools should never reach here (they execute within response stream)
        # If we get here, it's an error
        return {
            "call_id": tool_call.call_id,
            "tool_name": tool_call.tool_name,
            "content": f"Unknown tool `{tool_call.tool_name}` was called.",
        }

    async def _create_turn_streaming(
        self,
        messages: list[dict[str, Any]],
        session_id: str,
    ) -> AsyncIterator[AgentStreamChunk]:
        await self.initialize()

        # Generate turn_id
        turn_id = f"turn_{uuid4().hex[:12]}"

        # Create synthesizer
        synthesizer = TurnEventSynthesizer(session_id=session_id, turn_id=turn_id)

        request_headers = self.extra_headers

        # Main turn loop
        while True:
            # Create response stream
            raw_stream = await self.client.responses.create(
                model=self._model,
                instructions=self._instructions,
                conversation=session_id,
                input=messages,
                tools=self._tools,
                stream=True,
                extra_headers=request_headers,
            )

            # Process events
            function_calls_to_execute: list[ToolCall] = []  # Only client-side!

            for high_level_event in synthesizer.process_raw_stream(raw_stream):
                # Handle failures
                if isinstance(high_level_event, TurnFailed):
                    yield AgentStreamChunk(event=high_level_event)
                    return

                # Track function calls that need client execution
                if isinstance(high_level_event, StepCompleted):
                    if high_level_event.step_type == "inference":
                        result = high_level_event.result
                        if result.function_calls:  # Only client-side function calls
                            function_calls_to_execute = result.function_calls

                yield AgentStreamChunk(event=high_level_event)

            # If no client-side function calls, turn is done
            if not function_calls_to_execute:
                # Emit TurnCompleted
                response = synthesizer.last_response
                if not response:
                    raise RuntimeError("No response available")
                for event in synthesizer.finish_turn():
                    yield AgentStreamChunk(event=event, response=response)
                break

            # Execute client-side tools (emit tool execution step events)
            tool_step_id = f"{turn_id}_step_{synthesizer.step_counter}"
            synthesizer.step_counter += 1

            yield AgentStreamChunk(
                event=StepStarted(
                    step_id=tool_step_id,
                    step_type="tool_execution",
                    turn_id=turn_id,
                    metadata={"server_side": False},
                )
            )

            tool_responses = await self._run_tool_calls(function_calls_to_execute)

            yield AgentStreamChunk(
                event=StepCompleted(
                    step_id=tool_step_id,
                    step_type="tool_execution",
                    turn_id=turn_id,
                    result=ToolExecutionStepResult(
                        step_id=tool_step_id,
                        tool_calls=function_calls_to_execute,
                        tool_responses=tool_responses,
                    ),
                )
            )

            # Continue loop with tool outputs as input
            messages = [
                {
                    "type": "function_call_output",
                    "call_id": payload["call_id"],
                    "output": payload["content"],
                }
                for payload in tool_responses
            ]


class AgentUtils:
    @staticmethod
    def get_client_tools(
        tools: list[dict[str, Any] | ClientTool | Callable[..., Any]] | None,
    ) -> list[ClientTool]:
        if not tools:
            return []

        # Wrap any function in client_tool decorator
        tools = [client_tool(tool) if (callable(tool) and not isinstance(tool, ClientTool)) else tool for tool in tools]
        return [tool for tool in tools if isinstance(tool, ClientTool)]

    @staticmethod
    def get_tool_calls(chunk: AgentStreamChunk, tool_parser: ToolParser | None = None) -> list[ToolCall]:
        if not isinstance(chunk.event, StepProgress):
            return []

        delta = chunk.event.delta
        if not isinstance(delta, ToolCallIssuedDelta) or delta.tool_type != "function":
            return []

        tool_call = ToolCall(
            call_id=delta.call_id,
            tool_name=delta.tool_name,
            arguments=delta.arguments,
        )

        if tool_parser:
            completion = CompletionMessage(
                role="assistant",
                content="",
                tool_calls=[tool_call],
                stop_reason="end_of_turn",
            )
            return tool_parser.get_tool_calls(completion)

        return [tool_call]

    @staticmethod
    def get_turn_id(chunk: AgentStreamChunk) -> str | None:
        return chunk.response.turn.turn_id if chunk.response else None

    @staticmethod
    def normalize_tools(
        tools: list[dict[str, Any] | ClientTool | Callable[..., Any]] | None,
    ) -> tuple[list[dict[str, Any]], list[ClientTool]]:
        """Convert all tools to API format dicts.

        Returns:
            - List of tool dicts for responses.create(tools=...)
            - List of ClientTool instances for client-side execution
        """
        if not tools:
            return [], []

        tool_dicts: list[dict[str, Any]] = []
        client_tool_instances: list[ClientTool] = []

        for tool in tools:
            # Convert callable to ClientTool
            if callable(tool) and not isinstance(tool, ClientTool):
                tool = client_tool(tool)

            if isinstance(tool, ClientTool):
                # Convert ClientTool to function tool dict
                tool_def = tool.get_tool_definition()
                tool_dicts.append(tool_def)
                client_tool_instances.append(tool)
            elif isinstance(tool, dict):
                # Server-side tool dict (file_search, web_search, etc.)
                tool_dicts.append(tool)  # type: ignore[arg-type]
            else:
                raise TypeError(f"Unsupported tool type: {type(tool)!r}")

        return tool_dicts, client_tool_instances
