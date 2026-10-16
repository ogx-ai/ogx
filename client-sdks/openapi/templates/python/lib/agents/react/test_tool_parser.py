# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import json

from ..types import CompletionMessage
from .tool_parser import Action, ReActOutput, ReActToolParser


def _completion_message(react_output: ReActOutput) -> CompletionMessage:
    return CompletionMessage(
        role="assistant",
        content=react_output.model_dump_json(),
        tool_calls=[],
        stop_reason="stop",
    )


def test_get_tool_calls_returns_call_for_zero_argument_tool():
    """A valid action with no parameters must still produce a ToolCall.

    Regression test: `tool_params` is an empty list for a zero-argument tool,
    which is falsy in Python. The old `if tool_name and tool_params:` check
    silently dropped the tool call in this case.
    """
    output = _completion_message(
        ReActOutput(
            thought="I should list the files",
            action=Action(tool_name="list_files", tool_params=[]),
            answer=None,
        )
    )

    tool_calls = ReActToolParser().get_tool_calls(output)

    assert len(tool_calls) == 1
    assert tool_calls[0].tool_name == "list_files"
    assert json.loads(tool_calls[0].arguments) == {}


def test_get_tool_calls_returns_call_with_arguments():
    output = _completion_message(
        ReActOutput(
            thought="I should search",
            action=Action(
                tool_name="search",
                tool_params=[{"name": "query", "value": "ogx"}],
            ),
            answer=None,
        )
    )

    tool_calls = ReActToolParser().get_tool_calls(output)

    assert len(tool_calls) == 1
    assert tool_calls[0].tool_name == "search"
    assert json.loads(tool_calls[0].arguments) == {"query": "ogx"}


def test_get_tool_calls_returns_empty_when_answer_present():
    output = _completion_message(
        ReActOutput(thought="Done", action=None, answer="42")
    )

    assert ReActToolParser().get_tool_calls(output) == []


def test_get_tool_calls_returns_empty_when_no_action_or_answer():
    output = _completion_message(
        ReActOutput(thought="Thinking", action=None, answer=None)
    )

    assert ReActToolParser().get_tool_calls(output) == []


def test_get_tool_calls_returns_empty_on_invalid_json():
    output = CompletionMessage(
        role="assistant",
        content="not valid json",
        tool_calls=[],
        stop_reason="stop",
    )

    assert ReActToolParser().get_tool_calls(output) == []
