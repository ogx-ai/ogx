# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Truncation utilities for reactive context-length handling.

When ``truncation="auto"`` is set on a /v1/responses request, the inference loop
attempts to run normally. If the provider returns a context-length exceeded error,
an ``_is_context_length_error`` check triggers a turn-drop: the oldest semantic
turn (group of user / assistant / tool messages) is removed from the message list,
and inference is retried from scratch.

A "turn" is a sequence starting with a user message and extending through all
consequent assistant messages and tool-result messages until the next user message
or the end of history.  System and developer messages are never dropped.
"""

from typing import Any

from ogx_api.inference import OpenAIMessageParam


class _TurnGroups:
    """Result of :func:`build_turn_groups`: protected indices + sequential turn groups."""

    def __init__(self, protected: list[int], turns_turns: list[list[int]]) -> None:
        self.protected = protected
        self.turns_turns = turns_turns


def drop_oldest_turn(messages: list[OpenAIMessageParam], groups: _TurnGroups) -> list[OpenAIMessageParam]:
    """Remove the oldest semantic turn from ``messages`` and return the truncated list.

    Returns ``messages`` unmodified when there are no droppable turns (all messages
    are protected or the list is empty).
    """
    if not groups.turns_turns:
        return messages
    oldest_turn = groups.turns_turns.pop(0)
    dropped_indices: set[int] = set(oldest_turn)
    return [m for i, m in enumerate(messages) if i not in dropped_indices]


def _msg_field(msg: OpenAIMessageParam, field: str) -> Any:
    """Safely get a field from a Pydantic model or a plain dict."""
    get = getattr(msg, field, None)
    if get is not None:
        return get
    if isinstance(msg, dict):
        return msg.get(field)
    return None


def build_turn_groups(messages: list[OpenAIMessageParam]) -> _TurnGroups:
    """Aggregate messages into semantic turn groups for dropping.

    A "turn" starts with a user message and includes all subsequent assistant
    messages and tool results until the next user message or end of history.

    System / developer messages are never dropped (they stay in the ``protected``
    list).

    Returns a ``_TurnGroups`` with ``protected`` (indices never to drop) and
    ``turns_turns`` (sequential groups of indices representing semantic turns).
    """
    protected: list[int] = []
    turns_turns: list[list[int]] = []
    current_turn: list[int] = []
    for i, msg in enumerate(messages):
        role = _msg_field(msg, "role")
        if role in ("system", "developer"):
            protected.append(i)
            continue
        if role == "tool":
            continue
        if role == "user" and current_turn:
            turns_turns.append(current_turn)
            current_turn = []
        current_turn.append(i)
        # If next message is a tool result, group it into this turn
        if i + 1 < len(messages) and _msg_field(messages[i + 1], "role") == "tool":
            j = i + 1
            while j < len(messages) and _msg_field(messages[j], "role") == "tool":
                current_turn.append(j)
                j += 1
    if current_turn:
        turns_turns.append(current_turn)
    return _TurnGroups(protected, turns_turns)


def is_context_length_error(exc: Exception) -> bool:
    """Detect context-length exceeded from any OpenAI-compatible provider.

    Checks ``exc.body`` for patterns like "context length", "maximum context",
    "too many tokens", etc.
    """
    body = getattr(exc, "body", None)
    if not isinstance(body, dict):
        return False

    error_obj = body.get("error")
    if isinstance(error_obj, dict):
        msg = str(error_obj.get("message", ""))
        error_type = str(error_obj.get("type", ""))
        error_code = str(error_obj.get("code", ""))
    else:
        msg = str(body.get("message", ""))
        error_type = str(body.get("type", ""))
        error_code = str(body.get("code", ""))

    msg_lower = msg.lower()
    if any(kw in msg_lower for kw in ("context length", "maximum context", "too many tokens", "input length")):
        return True
    if any(kw in error_code.lower() for kw in ("context-window-exceeded", "context_length_exceeded")):
        return True
    if "input_text" in error_type.lower():
        return True
    return False
