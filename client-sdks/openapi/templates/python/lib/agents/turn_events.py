# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""High-level turn and step events for agent interactions.

This module defines the semantic event model that wraps the lower-level
responses API stream events. It provides a turn/step conceptual model that
makes agent interactions easier to understand and work with.

Key concepts:
- Turn: A complete interaction loop that may span multiple responses
- Step: A distinct phase within a turn (inference or tool_execution)
- Delta: Incremental updates during step execution
- Result: Complete output when a step finishes
"""

from dataclasses import dataclass
from typing import Any, Literal

from .types import ToolCall

__all__ = [
    "TurnStarted",
    "TurnCompleted",
    "TurnFailed",
    "StepStarted",
    "StepProgress",
    "StepCompleted",
    "TextDelta",
    "ToolCallIssuedDelta",
    "ToolCallDelta",
    "ToolCallCompletedDelta",
    "StepDelta",
    "InferenceStepResult",
    "ToolExecutionStepResult",
    "StepResult",
    "AgentEvent",
    "AgentStreamChunk",
]


# ============= Turn-Level Events =============


@dataclass
class TurnStarted:
    """Emitted when agent begins processing user input.

    This marks the beginning of a complete interaction cycle that may
    involve multiple inference steps and tool executions.
    """

    turn_id: str
    session_id: str
    event_type: Literal["turn_started"] = "turn_started"


@dataclass
class TurnCompleted:
    """Emitted when agent finishes with final answer.

    This marks the end of a turn when the model has produced a final
    response without any pending client-side tool calls.
    """

    turn_id: str
    session_id: str
    final_text: str
    response_ids: list[str]  # All response IDs involved in this turn
    num_steps: int
    event_type: Literal["turn_completed"] = "turn_completed"


@dataclass
class TurnFailed:
    """Emitted if turn processing fails.

    This indicates an unrecoverable error during turn processing.
    """

    turn_id: str
    session_id: str
    error_message: str
    event_type: Literal["turn_failed"] = "turn_failed"


# ============= Step-Level Events =============


@dataclass
class StepStarted:
    """Emitted when a distinct work phase begins.

    Steps represent distinct phases of work within a turn:
    - inference: Model thinking/generation (deciding what to do)
    - tool_execution: Tool execution (server-side or client-side)
    """

    step_id: str
    step_type: Literal["inference", "tool_execution"]
    turn_id: str
    event_type: Literal["step_started"] = "step_started"
    metadata: dict[str, Any] | None = None  # e.g., {"server_side": True/False, "tool_type": "file_search"}


# ============= Progress Delta Types =============


@dataclass
class TextDelta:
    """Incremental text during inference.

    Emitted as the model generates text token by token.
    """

    text: str
    delta_type: Literal["text"] = "text"


@dataclass
class ToolCallIssuedDelta:
    """Model initiates a tool call (client or server-side).

    This is emitted when the model decides to call a tool. The tool_type
    field indicates whether this is:
    - "function": Client-side tool requiring client execution
    - Other types: Server-side tools executed within the response
    """

    call_id: str
    tool_type: Literal["function", "file_search", "web_search", "mcp_call", "mcp_list_tools", "memory_retrieval"]
    tool_name: str
    arguments: str  # JSON string
    delta_type: Literal["tool_call_issued"] = "tool_call_issued"


@dataclass
class ToolCallDelta:
    """Incremental tool call arguments (streaming).

    Emitted as the model streams tool call arguments. The arguments
    are accumulated over multiple deltas to form the complete JSON.
    """

    call_id: str
    arguments_delta: str
    delta_type: Literal["tool_call_delta"] = "tool_call_delta"


@dataclass
class ToolCallCompletedDelta:
    """Server-side tool execution completed.

    Emitted when a server-side tool (file_search, web_search, etc.)
    finishes execution. The result field contains the tool output.

    Note: Client-side function tools do NOT emit this event; instead
    they trigger a separate tool_execution step.
    """

    call_id: str
    tool_type: Literal["file_search", "web_search", "mcp_call", "mcp_list_tools", "memory_retrieval"]
    tool_name: str
    result: Any  # Tool execution result from server
    delta_type: Literal["tool_call_completed"] = "tool_call_completed"


# Union of all delta types
StepDelta = TextDelta | ToolCallIssuedDelta | ToolCallDelta | ToolCallCompletedDelta


@dataclass
class StepProgress:
    """Emitted during step execution with streaming updates.

    Progress events provide real-time updates as a step executes,
    including text deltas and tool call information.
    """

    step_id: str
    step_type: Literal["inference", "tool_execution"]
    turn_id: str
    delta: StepDelta
    event_type: Literal["step_progress"] = "step_progress"


# ============= Step Result Types =============


@dataclass
class InferenceStepResult:
    """Complete inference step output.

    This contains the final accumulated state after an inference step
    completes. It separates client-side function calls (which need
    client execution) from server-side tool executions (which are
    included for logging/reference only).
    """

    step_id: str
    response_id: str
    text_content: str
    function_calls: list[ToolCall]  # Client-side function calls that need execution
    server_tool_executions: list[dict[str, Any]]  # Server-side tool calls (for reference/logging)
    stop_reason: str


@dataclass
class ToolExecutionStepResult:
    """Complete tool execution step output (client-side only).

    This contains the results of executing client-side function tools.
    These results will be fed back to the model in the next inference step.
    """

    step_id: str
    tool_calls: list[ToolCall]  # Function calls executed
    tool_responses: list[dict[str, Any]]  # Normalized responses


# Union of all result types
StepResult = InferenceStepResult | ToolExecutionStepResult


@dataclass
class StepCompleted:
    """Emitted when a step finishes.

    This provides the complete result of the step execution, including
    all accumulated data and final state.
    """

    step_id: str
    step_type: Literal["inference", "tool_execution"]
    turn_id: str
    result: StepResult
    event_type: Literal["step_completed"] = "step_completed"


# ============= Unified Event Type =============


# Union of all event types
AgentEvent = TurnStarted | StepStarted | StepProgress | StepCompleted | TurnCompleted | TurnFailed


@dataclass
class AgentStreamChunk:
    """What the agent yields to users.

    This is the top-level container for streaming events. Each chunk
    contains a high-level event (turn or step) and optionally the
    final response payload when the turn completes.

    Usage:
        for chunk in agent.create_turn(messages, session_id, stream=True):
            if isinstance(chunk.event, StepProgress):
                if isinstance(chunk.event.delta, TextDelta):
                    print(chunk.event.delta.text, end="")
            elif isinstance(chunk.event, TurnCompleted):
                print(f"\\nDone! Response: {chunk.response}")
    """

    event: AgentEvent
    response: Any | None = None  # Only set on TurnCompleted
