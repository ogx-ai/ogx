# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Event logger for agent interactions.

This module provides a simple logger that converts agent stream events
into human-readable printable strings for console output.
"""

from collections.abc import Iterator

from .turn_events import (
    AgentStreamChunk,
    StepCompleted,
    StepProgress,
    StepStarted,
    TextDelta,
    ToolCallDelta,
    ToolCallIssuedDelta,
    TurnCompleted,
    TurnFailed,
    TurnStarted,
)

__all__ = ["AgentEventLogger", "EventLogger"]


class AgentEventLogger:
    """Logger for agent events with turn/step semantics.

    This logger converts high-level agent events into printable strings
    that can be displayed to users. It handles:
    - Turn lifecycle events
    - Step boundaries (inference, tool execution)
    - Streaming content (text, tool calls)
    - Server-side and client-side tool execution

    Usage:
        logger = AgentEventLogger()
        for chunk in agent.create_turn(...):
            for printable in logger.log([chunk]):
                print(printable, end="", flush=True)
    """

    def log(self, event_generator: Iterator[AgentStreamChunk]) -> Iterator[str]:
        """Generate printable strings from agent stream chunks.

        Args:
            event_generator: Iterator of AgentStreamChunk objects

        Yields:
            Printable string fragments
        """
        for chunk in event_generator:
            event = chunk.event

            if isinstance(event, TurnStarted):
                # Optionally log turn start (commented out to reduce noise)
                # yield f"[Turn {event.turn_id}]\n"
                pass

            elif isinstance(event, StepStarted):
                if event.step_type == "inference":
                    # Indicate model is thinking (no newline)
                    yield "🤔 "
                elif event.step_type == "tool_execution":
                    # Indicate tools are executing
                    server_side = event.metadata and event.metadata.get("server_side", False)
                    if server_side:
                        tool_type = event.metadata.get("tool_type", "tool")
                        yield f"\n🔧 Executing {tool_type} (server-side)...\n"
                    else:
                        yield "\n🔧 Executing function tools (client-side)...\n"

            elif isinstance(event, StepProgress):
                if event.step_type == "inference":
                    if isinstance(event.delta, TextDelta):
                        # Stream text as it comes
                        yield event.delta.text

                    elif isinstance(event.delta, ToolCallIssuedDelta):
                        # Log client-side function calls (server-side handled as separate tool_execution steps)
                        if event.delta.tool_type == "function":
                            # Client-side function call
                            yield f"\n📞 Function call: {event.delta.tool_name}({event.delta.arguments})"

                    elif isinstance(event.delta, ToolCallDelta):
                        # Optionally stream tool arguments (can be noisy, so commented out)
                        # yield event.delta.arguments_delta
                        pass

                elif event.step_type == "tool_execution":
                    # Handle tool execution progress (for server-side tools)
                    if isinstance(event.delta, ToolCallIssuedDelta):
                        # Don't log again, already logged at StepStarted
                        pass
                    elif isinstance(event.delta, ToolCallDelta):
                        # Optionally log argument streaming
                        pass

            elif isinstance(event, StepCompleted):
                if event.step_type == "inference":
                    result = event.result
                    # Server-side tools already logged during progress
                    if not result.function_calls:
                        # End of inference with no function calls
                        yield "\n"

                elif event.step_type == "tool_execution":
                    # Log client-side tool execution results
                    result = event.result
                    for resp in result.tool_responses:
                        tool_name = resp.get("tool_name", "unknown")
                        content = resp.get("content", "")
                        # Truncate long responses for readability
                        if isinstance(content, str) and len(content) > 100:
                            content = content[:100] + "..."
                        yield f"  → {tool_name}: {content}\n"

            elif isinstance(event, TurnCompleted):
                # Optionally log turn completion (commented out to reduce noise)
                # yield f"\n[Completed in {event.num_steps} steps]\n"
                pass

            elif isinstance(event, TurnFailed):
                # Always log failures
                yield f"\n❌ Turn failed: {event.error_message}\n"


# Alias for backwards compatibility
EventLogger = AgentEventLogger
