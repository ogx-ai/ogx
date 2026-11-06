# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from __future__ import annotations

from ogx.providers.inline.responses.builtin.responses.truncation import (
    _TurnGroups,
    build_turn_groups,
    drop_oldest_turn,
    is_context_length_error,
)
from ogx_api.inference import (
    OpenAIAssistantMessageParam,
    OpenAIDeveloperMessageParam,
    OpenAIMessageParam,
    OpenAISystemMessageParam,
    OpenAIToolMessageParam,
    OpenAIUserMessageParam,
)


class TestBuildTurnGroups:
    """Test build_turn_groups message-to-turn aggregation."""

    def test_empty_messages(self) -> None:
        groups: _TurnGroups = build_turn_groups([])
        assert groups.protected == []
        assert groups.turns_turns == []

    def test_single_user_message(self) -> None:
        messages: list[OpenAIMessageParam] = [
            OpenAIUserMessageParam(role="user", content="Hello"),
        ]
        groups = build_turn_groups(messages)
        assert groups.protected == []
        assert groups.turns_turns == [[0]]

    def test_user_assistant_pair(self) -> None:
        messages: list[OpenAIMessageParam] = [
            OpenAIUserMessageParam(role="user", content="What is 2+2?"),
            OpenAIAssistantMessageParam(role="assistant", content="4"),
        ]
        groups = build_turn_groups(messages)
        assert groups.protected == []
        assert groups.turns_turns == [[0, 1]]

    def test_two_user_turns(self) -> None:
        messages: list[OpenAIMessageParam] = [
            OpenAIUserMessageParam(role="user", content="First question"),
            OpenAIAssistantMessageParam(role="assistant", content="First answer"),
            OpenAIUserMessageParam(role="user", content="Second question"),
            OpenAIAssistantMessageParam(role="assistant", content="Second answer"),
        ]
        groups = build_turn_groups(messages)
        assert groups.protected == []
        assert groups.turns_turns == [[0, 1], [2, 3]]

    def test_system_messages_protected(self) -> None:
        messages: list[OpenAIMessageParam] = [
            OpenAISystemMessageParam(role="system", content="You are helpful"),
            OpenAIUserMessageParam(role="user", content="Tell me a joke"),
            OpenAIAssistantMessageParam(role="assistant", content="Why did..."),
        ]
        groups = build_turn_groups(messages)
        assert groups.protected == [0]
        assert groups.turns_turns == [[1, 2]]

    def test_developer_messages_protected(self) -> None:
        messages: list[OpenAIMessageParam] = [
            OpenAIDeveloperMessageParam(content="Developer instructions"),
            OpenAIUserMessageParam(role="user", content="Hello"),
        ]
        groups = build_turn_groups(messages)
        assert groups.protected == [0]
        assert groups.turns_turns == [[1]]

    def test_system_messages_in_middle(self) -> None:
        messages: list[OpenAIMessageParam] = [
            OpenAIUserMessageParam(role="user", content="First question"),
            OpenAIAssistantMessageParam(role="assistant", content="First answer"),
            OpenAISystemMessageParam(role="system", content="System reminder"),
            OpenAIUserMessageParam(role="user", content="Second question"),
            OpenAIAssistantMessageParam(role="assistant", content="Second answer"),
        ]
        groups = build_turn_groups(messages)
        assert groups.protected == [2]
        assert groups.turns_turns == [[0, 1], [3, 4]]

    def test_tool_results_grouped_with_assistant(self) -> None:
        messages: list[OpenAIMessageParam] = [
            OpenAIUserMessageParam(role="user", content="Execute action"),
            OpenAIAssistantMessageParam(
                role="assistant",
                content="",
                tool_calls=[{"id": "tc1", "function": {"name": "calc", "arguments": "{}"}}],
            ),
            OpenAIToolMessageParam(role="tool", content="result: 42", tool_call_id="tc1"),
            OpenAIAssistantMessageParam(role="assistant", content="Done"),
        ]
        groups = build_turn_groups(messages)
        # user, assistant with tool_calls, tool result, and subsequent assistant content
        # are all in one turn (the lookahead groups them together)
        assert groups.turns_turns == [[0, 1, 2, 3]]

    def test_multiple_tool_results_grouped(self) -> None:
        messages: list[OpenAIMessageParam] = [
            OpenAIUserMessageParam(role="user", content="Do stuff"),
            OpenAIAssistantMessageParam(role="assistant", content=""),
            OpenAIToolMessageParam(role="tool", content="result 1", tool_call_id="tc1"),
            OpenAIToolMessageParam(role="tool", content="result 2", tool_call_id="tc2"),
            OpenAIAssistantMessageParam(role="assistant", content="All done"),
        ]
        groups = build_turn_groups(messages)
        # user, assistant, two tool results, and subsequent assistant are all in one turn
        assert groups.turns_turns == [[0, 1, 2, 3, 4]]

    def test_drop_oldest_turn(self) -> None:
        original: list[OpenAIMessageParam] = [
            OpenAIUserMessageParam(role="user", content="Old question A"),
            OpenAIAssistantMessageParam(role="assistant", content="Old answer A"),
            OpenAIUserMessageParam(role="user", content="Old question B"),
            OpenAIAssistantMessageParam(role="assistant", content="Old answer B"),
            OpenAIUserMessageParam(role="user", content="New question"),
        ]
        groups = build_turn_groups(original)
        assert groups.turns_turns == [[0, 1], [2, 3], [4]]

        truncated = drop_oldest_turn(original, groups)
        assert len(truncated) == 3
        assert truncated[0].content == "Old question B"
        assert truncated[1].content == "Old answer B"
        assert truncated[2].content == "New question"

        # The original list is untouched
        assert len(original) == 5

    def test_drop_no_turns_returns_original(self) -> None:
        messages: list[OpenAIMessageParam] = [
            OpenAISystemMessageParam(role="system", content="You are helpful"),
            OpenAIDeveloperMessageParam(content="Developer instructions"),
        ]
        groups = build_turn_groups(messages)
        assert groups.turns_turns == []

        result = drop_oldest_turn(messages, groups)
        assert result is messages
        assert len(result) == 2

    def test_drop_empty_messages(self) -> None:
        groups = build_turn_groups([])
        result = drop_oldest_turn([], groups)
        assert result == []

    def test_turn_groups_with_system_and_tool_results(self) -> None:
        messages: list[OpenAIMessageParam] = [
            OpenAISystemMessageParam(role="system", content="You are helpful"),
            OpenAIUserMessageParam(role="user", content="Execute command"),
            OpenAIAssistantMessageParam(role="assistant", content=""),
            OpenAIToolMessageParam(role="tool", content="command output", tool_call_id="tc1"),
            OpenAIAssistantMessageParam(role="assistant", content="Done"),
            OpenAIUserMessageParam(role="user", content="Next question"),
        ]
        groups = build_turn_groups(messages)
        assert groups.protected == [0]
        assert groups.turns_turns == [[1, 2, 3, 4], [5]]


class TestIsContextLengthError:
    """Test is_context_length_error detection for various providers."""

    def test_openai_format_with_error_obj(self) -> None:
        exc = Exception("Context length exceeded")
        exc.body = {
            "error": {
                "code": 400,
                "message": "This model's maximum context length is 128000. Input has 200000.",
                "type": "BadRequestError",
                "param": None,
            }
        }
        assert is_context_length_error(exc)

    def test_openai_format_direct(self) -> None:
        exc = Exception("Context length exceeded")
        exc.body = {
            "code": 400,
            "message": "Input length is too long.",
            "type": "invalid_request_error",
        }
        assert is_context_length_error(exc)

    def test_context_length_keywords_match(self) -> None:
        exc = Exception("error")
        exc.body = {"error": {"message": "Context length exceeded", "type": "error", "code": 400}}
        assert is_context_length_error(exc)

        exc.body = {"error": {"message": "Maximum context length exceeded", "type": "error", "code": 400}}
        assert is_context_length_error(exc)

        exc.body = {"error": {"message": "Too many tokens in input", "type": "error", "code": 400}}
        assert is_context_length_error(exc)

        exc.body = {"error": {"message": "Input length exceeds limit", "type": "error", "code": 400}}
        assert is_context_length_error(exc)

    def test_context_window_exceeded_code(self) -> None:
        exc = Exception("error")
        exc.body = {"error": {"message": "Context window exceeded", "type": "error", "code": "context-window-exceeded"}}
        assert is_context_length_error(exc)

        exc.body = {
            "error": {"message": "Context length exceeded", "type": "TypeClassError", "code": "context_length_exceeded"}
        }
        assert is_context_length_error(exc)

    def test_input_text_in_type(self) -> None:
        exc = Exception("error")
        exc.body = {"error": {"message": "Some error", "type": "input_text_error", "code": "400"}}
        assert is_context_length_error(exc)

    def test_not_context_error_non_dict_body(self) -> None:
        exc = Exception("network error")
        exc.body = "Connection refused"
        assert not is_context_length_error(exc)

    def test_not_context_error_no_body(self) -> None:
        exc = Exception("some error")
        assert not is_context_length_error(exc)

    def test_not_context_error_other_error(self) -> None:
        exc = Exception("rate limit")
        exc.body = {"error": {"message": "Rate limit exceeded", "type": "RateLimitError", "code": 429}}
        assert not is_context_length_error(exc)

    def test_not_context_error_auth_error(self) -> None:
        exc = Exception("auth failed")
        exc.body = {"error": {"message": "Invalid API key", "type": "AuthenticationError", "code": 401}}
        assert not is_context_length_error(exc)


class TestTurnGroupsDataStructure:
    """Test _TurnGroups data structure."""

    def test_tg_basic(self) -> None:
        tg = _TurnGroups(protected=[0, 3], turns_turns=[[1, 2], [4, 5]])
        assert tg.protected == [0, 3]
        assert tg.turns_turns == [[1, 2], [4, 5]]

    def test_tg_empty(self) -> None:
        tg = _TurnGroups(protected=[], turns_turns=[])
        assert tg.protected == []
        assert tg.turns_turns == []
