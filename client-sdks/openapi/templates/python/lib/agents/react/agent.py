# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
import logging
from collections.abc import Callable, Mapping
from typing import Any

from ...._types import Headers
from ..agent import Agent, AgentUtils
from ..client_tool import ClientTool
from ..tool_parser import ToolParser
from .prompts import DEFAULT_REACT_AGENT_SYSTEM_PROMPT_TEMPLATE
from .tool_parser import ReActToolParser

logger = logging.getLogger(__name__)


def _tool_definition_from_mapping(tool: Mapping[str, Any]) -> dict[str, Any]:
    name = tool.get("name") or tool.get("identifier") or tool.get("tool_name") or tool.get("type") or "tool"
    description = tool.get("description") or tool.get("summary") or ""
    parameters = tool.get("parameters") or tool.get("input_schema") or {}
    return {
        "name": str(name),
        "description": str(description),
        "input_schema": parameters,
    }


def _collect_tool_definitions(
    server_tools: tuple[Mapping[str, Any], ...],
    client_tools: tuple[ClientTool, ...],
) -> list[dict[str, Any]]:
    tool_defs = [_tool_definition_from_mapping(tool) for tool in server_tools]
    tool_defs.extend(
        {
            "name": tool.get_name(),
            "description": tool.get_description(),
            "input_schema": tool.get_input_schema(),
        }
        for tool in client_tools
    )
    return tool_defs


def get_default_react_instructions(tool_defs: list[dict[str, Any]]) -> str:
    tool_names = ", ".join([definition["name"] for definition in tool_defs])
    tool_descriptions = "\n".join(
        [
            f"- {definition['name']}: {definition['description'] or definition['input_schema']}"
            for definition in tool_defs
        ]
    )
    instruction = DEFAULT_REACT_AGENT_SYSTEM_PROMPT_TEMPLATE.replace("<<tool_names>>", tool_names).replace(
        "<<tool_descriptions>>", tool_descriptions
    )
    return instruction


class ReActAgent(Agent):
    def __init__(
        self,
        client: Any,
        *,
        model: str,
        tools: list[dict[str, Any] | ClientTool | Callable[..., Any]] | None = None,
        tool_parser: ToolParser | None = None,
        instructions: str | None = None,
        extra_headers: Headers | None = None,
        json_response_format: bool = False,
    ):
        if json_response_format:
            logger.warning("`json_response_format` is deprecated and will be removed in a future release.")

        if tool_parser is None:
            tool_parser = ReActToolParser()

        tool_list = tools or []
        client_tool_instances = AgentUtils.get_client_tools(tool_list)
        server_tool_defs = tuple(tool for tool in tool_list if isinstance(tool, Mapping))

        if instructions is None:
            tool_definitions = _collect_tool_definitions(tuple(server_tool_defs), tuple(client_tool_instances))
            instructions = get_default_react_instructions(tool_definitions)

        super().__init__(
            client=client,
            model=model,
            instructions=instructions,
            tools=tool_list,
            tool_parser=tool_parser,
            extra_headers=extra_headers,
        )
