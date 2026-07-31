# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Lightweight agent-facing types that avoid ogx SDK dependencies."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, TypedDict


@dataclass
class ToolCall:
    """Minimal representation of an issued tool call."""

    call_id: str
    tool_name: str
    arguments: str


@dataclass
class ToolResponse:
    """Payload returned from executing a client-side tool."""

    call_id: str
    tool_name: str
    content: Any
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CompletionMessage:
    """Synthetic completion message mirroring the OpenAI Responses schema."""

    role: str
    content: Any
    tool_calls: list[ToolCall]
    stop_reason: str


Message = CompletionMessage


class ToolDefinition(TypedDict, total=False):
    """Definition object passed to the Responses API when registering tools."""

    type: str
    name: str
    description: str
    parameters: dict[str, Any]


class FunctionTool(Protocol):
    """Protocol describing the minimal surface area we expect from tools."""

    def get_name(self) -> str: ...

    def get_description(self) -> str: ...

    def get_input_schema(self) -> dict[str, Any]: ...
