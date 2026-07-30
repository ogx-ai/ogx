# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Translate Responses API stream events into structured turn events.

TurnEventSynthesizer keeps just enough state to expose turns and steps for
agents. It consumes the raw Responses API stream and emits the higher-level
events defined in ``turn_events.py`` without introducing an intermediate
low-level event layer.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from logging import getLogger
from typing import Any

from .turn_events import (
    AgentEvent,
    InferenceStepResult,
    StepCompleted,
    StepProgress,
    StepStarted,
    TextDelta,
    ToolCallDelta,
    ToolCallIssuedDelta,
    ToolExecutionStepResult,
    TurnCompleted,
    TurnFailed,
    TurnStarted,
)
from .types import ToolCall

logger = getLogger(__name__)


@dataclass
class _ToolCallState:
    call_id: str
    tool_name: str
    tool_type: str
    server_side: bool
    arguments: str = ""

    def update(self, *, delta: str | None = None, final: str | None = None) -> None:
        if final is not None:
            self.arguments = final or "{}"
        elif delta:
            self.arguments += delta

    def as_tool_call(self) -> ToolCall:
        payload = self.arguments or "{}"
        return ToolCall(call_id=self.call_id, tool_name=self.tool_name, arguments=payload)


class TurnEventSynthesizer:
    """Produce turn/step events directly from Responses API streaming events."""

    def __init__(self, session_id: str, turn_id: str):
        self.session_id = session_id
        self.turn_id = turn_id

        self.step_counter = 0
        self.current_step_id: str | None = None
        self.current_step_type: str | None = None

        self.current_response_id: str | None = None
        self.text_parts: list[str] = []
        self._function_call_ids: list[str] = []
        self._tool_calls: dict[str, _ToolCallState] = {}

        self.turn_started = False
        self.all_response_ids: list[str] = []
        self.last_response: Any | None = None

    # ------------------------------------------------------------------ helpers

    def _next_step_id(self) -> str:
        step_id = f"{self.turn_id}_step_{self.step_counter}"
        self.step_counter += 1
        return step_id

    def _maybe_emit_turn_started(self) -> Iterator[AgentEvent]:
        if not self.turn_started:
            self.turn_started = True
            yield TurnStarted(turn_id=self.turn_id, session_id=self.session_id)

    def _start_inference_step(
        self, *, response_id: str | None = None, reset_tool_state: bool = False
    ) -> Iterator[AgentEvent]:
        if response_id:
            self.current_response_id = response_id
        if reset_tool_state:
            self._tool_calls = {}
        self.text_parts = []
        self._function_call_ids = []
        self.current_step_id = self._next_step_id()
        self.current_step_type = "inference"
        yield StepStarted(step_id=self.current_step_id, step_type="inference", turn_id=self.turn_id)

    def _complete_inference_step(self, *, stop_reason: str, response_id: str | None = None) -> Iterator[AgentEvent]:
        if self.current_step_type != "inference":
            return
        step_id = self.current_step_id or self._next_step_id()
        resolved_response_id = response_id or self.current_response_id or ""
        self.current_response_id = resolved_response_id

        function_calls: list[ToolCall] = []
        for call_id in self._function_call_ids:
            state = self._tool_calls.get(call_id)
            if state is None:
                continue
            function_calls.append(state.as_tool_call())

        yield StepCompleted(
            step_id=step_id,
            step_type="inference",
            turn_id=self.turn_id,
            result=InferenceStepResult(
                step_id=step_id,
                response_id=resolved_response_id,
                text_content="".join(self.text_parts),
                function_calls=function_calls,
                server_tool_executions=[],
                stop_reason=stop_reason,
            ),
        )

        for call_id in list(self._function_call_ids):
            # Drop client-side call state once we hand it back to the agent.
            self._tool_calls.pop(call_id, None)
        self._function_call_ids = []
        self.text_parts = []
        self.current_step_id = None
        self.current_step_type = None

    def _start_tool_execution_step(self, call_state: _ToolCallState) -> Iterator[AgentEvent]:
        self.current_step_id = self._next_step_id()
        self.current_step_type = "tool_execution"
        yield StepStarted(
            step_id=self.current_step_id,
            step_type="tool_execution",
            turn_id=self.turn_id,
            metadata={"server_side": True, "tool_type": call_state.tool_type, "tool_name": call_state.tool_name},
        )
        yield StepProgress(
            step_id=self.current_step_id,
            step_type="tool_execution",
            turn_id=self.turn_id,
            delta=ToolCallIssuedDelta(
                call_id=call_state.call_id,
                tool_type=call_state.tool_type,  # type: ignore[arg-type]
                tool_name=call_state.tool_name,
                arguments=call_state.arguments or "{}",
            ),
        )

    def _complete_tool_execution_step(self, call_state: _ToolCallState) -> Iterator[AgentEvent]:
        if self.current_step_type != "tool_execution":
            return
        step_id = self.current_step_id or self._next_step_id()
        yield StepCompleted(
            step_id=step_id,
            step_type="tool_execution",
            turn_id=self.turn_id,
            result=ToolExecutionStepResult(
                step_id=step_id,
                tool_calls=[call_state.as_tool_call()],
                tool_responses=[],
            ),
        )
        self._tool_calls.pop(call_state.call_id, None)
        self.current_step_id = None
        self.current_step_type = None

    @staticmethod
    def _coerce_arguments(payload: Any) -> str | None:
        if payload is None:
            return None
        if isinstance(payload, str):
            return payload
        try:
            return json.dumps(payload)
        except Exception:  # pragma: no cover - defensive
            return str(payload)

    def _register_tool_call(
        self,
        *,
        call_id: str | None,
        tool_name: str,
        tool_type: str,
        arguments: str | None,
    ) -> _ToolCallState:
        resolved_call_id = call_id or f"{tool_name}_{len(self._tool_calls)}"
        state = _ToolCallState(
            call_id=resolved_call_id,
            tool_name=tool_name,
            tool_type=tool_type,
            server_side=tool_type != "function",
        )
        if arguments:
            state.update(final=arguments)
        self._tool_calls[resolved_call_id] = state
        return state

    def _classify_tool_type(self, tool_name: str) -> str:
        server_side_tools = {
            "file_search",
            "file_search_call",
            "knowledge_search",
            "web_search",
            "web_search_call",
            "query_from_memory",
            "mcp_call",
            "mcp_list_tools",
            "memory_retrieval",
        }

        if tool_name in server_side_tools:
            if tool_name in {"file_search_call", "knowledge_search"}:
                return "file_search"
            if tool_name == "web_search_call":
                return "web_search"
            return tool_name

        return "function"

    # ------------------------------------------------------------------ handlers

    def process_raw_stream(self, events: Iterable[Any]) -> Iterator[AgentEvent]:
        current_response_id: str | None = None

        for event in events:
            yield from self._maybe_emit_turn_started()

            response_id = getattr(event, "response_id", None)
            if response_id is None and hasattr(event, "response"):
                response = event.response
                response_id = getattr(response, "id", None)
            if response_id is not None:
                current_response_id = response_id

            event_type = getattr(event, "type", None)
            if not event_type:
                logger.debug("Unhandled stream event with no type: %r", event)
                continue

            if event_type == "response.in_progress":
                response = getattr(event, "response", None)
                response_id = getattr(response, "id", current_response_id)
                if response_id is None:
                    continue
                if not self.all_response_ids or self.all_response_ids[-1] != response_id:
                    self.all_response_ids.append(response_id)
                yield from self._start_inference_step(response_id=response_id, reset_tool_state=True)

            elif event_type == "response.output_text.delta":
                if self.current_step_type != "inference":
                    continue
                text = getattr(event, "delta", "") or ""
                if not text:
                    continue
                self.text_parts.append(text)
                yield StepProgress(
                    step_id=self.current_step_id or "",
                    step_type="inference",
                    turn_id=self.turn_id,
                    delta=TextDelta(text=text),
                )

            elif event_type == "response.output_text.done":
                # Text completions are tracked via the final StepCompleted event.
                continue

            elif event_type == "response.output_item.added":
                yield from self._handle_output_item_added(getattr(event, "item", None))

            elif event_type == "response.output_item.delta":
                yield from self._handle_output_item_delta(getattr(event, "delta", None))

            elif event_type == "response.output_item.done":
                yield from self._handle_output_item_done(getattr(event, "item", None))

            elif event_type == "response.completed":
                response = getattr(event, "response", None)
                if response is not None:
                    self.last_response = response
                stop_reason = "tool_calls" if self._function_call_ids else "end_of_turn"
                yield from self._complete_inference_step(stop_reason=stop_reason, response_id=current_response_id)

            elif event_type == "response.failed":
                response = getattr(event, "response", None)
                error_obj = getattr(response, "error", None)
                error_message = getattr(error_obj, "message", None) if error_obj else None
                yield TurnFailed(
                    turn_id=self.turn_id,
                    session_id=self.session_id,
                    error_message=error_message or "Unknown error",
                )

            else:  # pragma: no cover - depends on streaming responses
                # Allow unknown streaming events to pass silently; they are often ancillary metadata.
                continue

    def _handle_output_item_added(self, item: Any) -> Iterator[AgentEvent]:
        if item is None:
            return
        item_type = getattr(item, "type", None)
        if item_type is None:
            return

        if item_type == "message":
            # Messages mirror text deltas, nothing extra to emit.
            return

        if item_type in {
            "function_call",
            "web_search",
            "web_search_call",
            "mcp_call",
            "mcp_list_tools",
            "file_search_call",
        }:
            call_id = getattr(item, "call_id", None) or getattr(item, "id", None)
            tool_name = getattr(item, "name", None) or getattr(item, "type", "")
            arguments = self._coerce_arguments(getattr(item, "arguments", None))
            tool_type = self._classify_tool_type(tool_name if item_type == "function_call" else item_type)

            state = self._register_tool_call(
                call_id=call_id,
                tool_name=tool_name,
                tool_type=tool_type,
                arguments=arguments,
            )

            if state.server_side:
                yield from self._complete_inference_step(
                    stop_reason="server_tool_call", response_id=self.current_response_id
                )
                yield from self._start_tool_execution_step(state)
            else:
                if state.call_id not in self._function_call_ids:
                    self._function_call_ids.append(state.call_id)
                yield StepProgress(
                    step_id=self.current_step_id or "",
                    step_type="inference",
                    turn_id=self.turn_id,
                    delta=ToolCallIssuedDelta(
                        call_id=state.call_id,
                        tool_type="function",
                        tool_name=state.tool_name,
                        arguments=state.arguments or "{}",
                    ),
                )

    def _handle_output_item_delta(self, delta: Any) -> Iterator[AgentEvent]:
        if delta is None:
            return
        delta_type = getattr(delta, "type", None)
        if delta_type not in {
            "function_call",
            "web_search",
            "web_search_call",
            "mcp_call",
            "mcp_list_tools",
            "file_search_call",
        }:
            return

        call_id = getattr(delta, "call_id", None) or getattr(delta, "id", None)
        if call_id is None:
            return

        arguments_delta = getattr(delta, "arguments_delta", None)
        if arguments_delta is None:
            arguments_delta = getattr(delta, "arguments", None)
        if arguments_delta is None and isinstance(delta, dict):
            arguments_delta = delta.get("arguments_delta") or delta.get("arguments")
        if arguments_delta is None:
            return

        state = self._tool_calls.get(call_id)
        if state is None:
            return

        state.update(delta=arguments_delta)
        step_type = "tool_execution" if state.server_side else "inference"
        yield StepProgress(
            step_id=self.current_step_id or "",
            step_type=step_type,  # type: ignore[arg-type]
            turn_id=self.turn_id,
            delta=ToolCallDelta(call_id=state.call_id, arguments_delta=arguments_delta),
        )

    def _handle_output_item_done(self, item: Any) -> Iterator[AgentEvent]:
        if item is None:
            return

        item_type = getattr(item, "type", None)
        if item_type not in {
            "function_call",
            "web_search",
            "web_search_call",
            "mcp_call",
            "mcp_list_tools",
            "file_search_call",
        }:
            return

        call_id = getattr(item, "call_id", None) or getattr(item, "id", None)
        if call_id is None:
            return

        state = self._tool_calls.get(call_id)
        if state is None:
            return

        arguments = self._coerce_arguments(getattr(item, "arguments", None))
        if arguments:
            state.update(final=arguments)

        if state.server_side:
            yield from self._complete_tool_execution_step(state)
            # Start a fresh inference step so the model can continue reasoning.
            yield from self._start_inference_step()
        else:
            if call_id not in self._function_call_ids:
                self._function_call_ids.append(call_id)

    # ------------------------------------------------------------------ turn end

    def finish_turn(self) -> Iterator[AgentEvent]:
        if not self.last_response:
            raise RuntimeError("Cannot finish turn without a response")

        yield TurnCompleted(
            turn_id=self.turn_id,
            session_id=self.session_id,
            final_text=self.last_response.output_text,
            response_ids=self.all_response_ids,
            num_steps=self.step_counter,
        )
