# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

from ogx.providers.inline.responses.builtin.responses.streaming import _VALID_RESPONSE_ERROR_CODES


class StreamingValidator:
    """Helper class for validating streaming response events."""

    def __init__(self, chunks: list[Any]):
        self.chunks = chunks
        self.event_types = [chunk.type for chunk in chunks]

    def assert_basic_event_sequence(self):
        """Verify basic created -> completed event sequence."""
        assert len(self.chunks) >= 2, f"Expected at least 2 chunks (created + terminal), got {len(self.chunks)}"
        assert self.chunks[0].type == "response.created", (
            f"First chunk should be response.created, got {self.chunks[0].type}"
        )
        assert any(t in self.event_types for t in ["response.completed", "response.incomplete", "response.failed"]), (
            "Expected a terminal response event (completed, incomplete, or failed)"
        )

        terminal_types = ["response.completed", "response.incomplete", "response.failed"]
        terminal_indices = [self.event_types.index(t) for t in terminal_types if t in self.event_types]
        assert terminal_indices, "Expected at least one terminal event index"
        created_index = self.event_types.index("response.created")
        assert created_index < min(terminal_indices), "response.created should precede terminal events"

    def assert_response_consistency(self):
        """Verify response ID consistency across events."""
        response_ids = set()
        for chunk in self.chunks:
            if hasattr(chunk, "response_id"):
                response_ids.add(chunk.response_id)
            elif hasattr(chunk, "response") and hasattr(chunk.response, "id"):
                response_ids.add(chunk.response.id)

        assert len(response_ids) == 1, f"All events should reference the same response_id, found: {response_ids}"

    def assert_has_incremental_content(self):
        """Verify that content is delivered incrementally via delta events."""
        delta_events = [
            i for i, event_type in enumerate(self.event_types) if event_type == "response.output_text.delta"
        ]
        assert len(delta_events) > 0, "Expected delta events for true incremental streaming, but found none"

        # Verify delta events have content
        non_empty_deltas = 0
        delta_content_total = ""

        for delta_idx in delta_events:
            chunk = self.chunks[delta_idx]
            if hasattr(chunk, "delta") and chunk.delta:
                delta_content_total += chunk.delta
                non_empty_deltas += 1

        assert non_empty_deltas > 0, "Delta events found but none contain content"
        assert len(delta_content_total) > 0, "Delta events found but total delta content is empty"

        return delta_content_total

    def assert_content_quality(self, expected_content: str):
        """Verify the final response contains expected content."""
        final_chunk = self.chunks[-1]
        if hasattr(final_chunk, "response"):
            output_text = final_chunk.response.output_text.lower().strip()
            assert len(output_text) > 0, "Response should have content"
            assert expected_content.lower() in output_text, f"Expected '{expected_content}' in response"

    def assert_has_tool_calls(self):
        """Verify tool call streaming events are present."""
        # Check for tool call events
        delta_events = [
            chunk
            for chunk in self.chunks
            if chunk.type in ["response.function_call_arguments.delta", "response.mcp_call.arguments.delta"]
        ]
        done_events = [
            chunk
            for chunk in self.chunks
            if chunk.type in ["response.function_call_arguments.done", "response.mcp_call.arguments.done"]
        ]

        assert len(delta_events) > 0, f"Expected tool call delta events, got chunk types: {self.event_types}"
        assert len(done_events) > 0, f"Expected tool call done events, got chunk types: {self.event_types}"

        # Verify output item events
        item_added_events = [chunk for chunk in self.chunks if chunk.type == "response.output_item.added"]
        item_done_events = [chunk for chunk in self.chunks if chunk.type == "response.output_item.done"]

        assert len(item_added_events) > 0, (
            f"Expected response.output_item.added events, got chunk types: {self.event_types}"
        )
        assert len(item_done_events) > 0, (
            f"Expected response.output_item.done events, got chunk types: {self.event_types}"
        )

    def assert_has_mcp_events(self):
        """Verify MCP-specific streaming events are present."""
        # Tool execution progress events
        mcp_in_progress_events = [chunk for chunk in self.chunks if chunk.type == "response.mcp_call.in_progress"]
        mcp_completed_events = [chunk for chunk in self.chunks if chunk.type == "response.mcp_call.completed"]

        assert len(mcp_in_progress_events) > 0, (
            f"Expected response.mcp_call.in_progress events, got chunk types: {self.event_types}"
        )
        assert len(mcp_completed_events) > 0, (
            f"Expected response.mcp_call.completed events, got chunk types: {self.event_types}"
        )

        # MCP list tools events
        mcp_list_tools_in_progress_events = [
            chunk for chunk in self.chunks if chunk.type == "response.mcp_list_tools.in_progress"
        ]
        mcp_list_tools_completed_events = [
            chunk for chunk in self.chunks if chunk.type == "response.mcp_list_tools.completed"
        ]

        assert len(mcp_list_tools_in_progress_events) > 0, (
            f"Expected response.mcp_list_tools.in_progress events, got chunk types: {self.event_types}"
        )
        assert len(mcp_list_tools_completed_events) > 0, (
            f"Expected response.mcp_list_tools.completed events, got chunk types: {self.event_types}"
        )

    def assert_rich_streaming(self, min_chunks: int = 10):
        """Verify we have substantial streaming activity."""
        assert len(self.chunks) > min_chunks, (
            f"Expected rich streaming with many events, got only {len(self.chunks)} chunks"
        )

    def validate_event_structure(self):
        """Validate the structure of various event types."""
        for chunk in self.chunks:
            if chunk.type == "response.created":
                assert chunk.response.status == "in_progress"
            elif chunk.type == "response.in_progress":
                assert chunk.response.status == "in_progress"
                assert isinstance(chunk.sequence_number, int)
            elif chunk.type == "response.incomplete":
                assert chunk.response.status == "incomplete"
                assert isinstance(chunk.sequence_number, int)
            elif chunk.type == "response.failed":
                assert chunk.response.status == "failed"
                assert isinstance(chunk.sequence_number, int)
                assert chunk.response.error is not None
                assert chunk.response.error.code in _VALID_RESPONSE_ERROR_CODES, (
                    f"Error code '{chunk.response.error.code}' is not a valid Responses API error code"
                )
                assert chunk.response.error.message, "Error message should be non-empty"
            elif chunk.type == "response.completed":
                assert chunk.response.status == "completed"
            elif chunk.type in {"response.content_part.added", "response.content_part.done"}:
                assert chunk.item_id, "Content part events should have non-empty item_id"
                assert isinstance(chunk.content_index, int)
                assert isinstance(chunk.output_index, int)
                assert chunk.response_id, "Content part events should include response_id"
            elif hasattr(chunk, "item_id"):
                assert chunk.item_id, "Events with item_id should have non-empty item_id"
            elif hasattr(chunk, "sequence_number"):
                assert isinstance(chunk.sequence_number, int), "sequence_number should be an integer"
