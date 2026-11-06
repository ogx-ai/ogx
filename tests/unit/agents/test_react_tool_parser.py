# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import json
import sys
import uuid
from pathlib import Path

# Add python client sdk lib directory to sys.path
sdk_lib_path = Path(__file__).resolve().parents[3] / "client-sdks" / "openapi" / "templates" / "python" / "lib"
if str(sdk_lib_path) not in sys.path:
    sys.path.insert(0, str(sdk_lib_path))

from agents.react.tool_parser import Action, Param, ReActOutput, ReActToolParser
from agents.types import CompletionMessage


def test_tool_call_with_arguments():
    """Verify that ReActToolParser correctly extracts tool calls with arguments."""
    parser = ReActToolParser()
    message = CompletionMessage(
        role="assistant",
        content=ReActOutput(
            thought="I need to inspect the file content.",
            action=Action(
                tool_name="read_file",
                tool_params=[
                    Param(name="path", value="README.md"),
                    Param(name="max_lines", value=50),
                ],
            ),
            answer=None,
        ).model_dump_json(),
        tool_calls=[],
        stop_reason="end_turn",
    )

    tool_calls = parser.get_tool_calls(message)
    assert len(tool_calls) == 1
    call = tool_calls[0]
    assert call.tool_name == "read_file"
    args = json.loads(call.arguments)
    assert args == {"path": "README.md", "max_lines": 50}
    # Verify call_id is a valid UUID
    uuid.UUID(call.call_id)


def test_tool_call_with_zero_arguments():
    """Verify that ReActToolParser extracts tool calls for zero-argument tools."""
    parser = ReActToolParser()
    message = CompletionMessage(
        role="assistant",
        content=ReActOutput(
            thought="I should list the available files in this directory.",
            action=Action(
                tool_name="list_files",
                tool_params=[],
            ),
            answer=None,
        ).model_dump_json(),
        tool_calls=[],
        stop_reason="end_turn",
    )

    tool_calls = parser.get_tool_calls(message)
    assert len(tool_calls) == 1, "Zero-argument tool call should not be dropped"
    call = tool_calls[0]
    assert call.tool_name == "list_files"
    args = json.loads(call.arguments)
    assert args == {}
    uuid.UUID(call.call_id)


def test_no_tool_call_when_answer_provided():
    """Verify that no tool call is generated when the model outputs a final answer."""
    parser = ReActToolParser()
    message = CompletionMessage(
        role="assistant",
        content=ReActOutput(
            thought="I now have enough information to answer the user.",
            action=None,
            answer="The repository contains 12 active modules.",
        ).model_dump_json(),
        tool_calls=[],
        stop_reason="end_turn",
    )

    tool_calls = parser.get_tool_calls(message)
    assert tool_calls == []


def test_no_tool_call_when_answer_precedes_action():
    """Verify that an answer takes precedence and prevents tool calls."""
    parser = ReActToolParser()
    message = CompletionMessage(
        role="assistant",
        content=ReActOutput(
            thought="Done.",
            action=Action(tool_name="list_files", tool_params=[]),
            answer="Here is the final response.",
        ).model_dump_json(),
        tool_calls=[],
        stop_reason="end_turn",
    )

    tool_calls = parser.get_tool_calls(message)
    assert tool_calls == []


def test_no_tool_call_when_action_is_none():
    """Verify that an empty thought with no action returns no tool calls."""
    parser = ReActToolParser()
    message = CompletionMessage(
        role="assistant",
        content=ReActOutput(
            thought="Just thinking...",
            action=None,
            answer=None,
        ).model_dump_json(),
        tool_calls=[],
        stop_reason="end_turn",
    )

    tool_calls = parser.get_tool_calls(message)
    assert tool_calls == []


def test_invalid_json_content():
    """Verify that malformed JSON response is caught and returns empty tool calls."""
    parser = ReActToolParser()
    message = CompletionMessage(
        role="assistant",
        content="This is not valid JSON at all.",
        tool_calls=[],
        stop_reason="end_turn",
    )

    tool_calls = parser.get_tool_calls(message)
    assert tool_calls == []
