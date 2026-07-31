# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import json
import uuid

from pydantic import BaseModel, ValidationError

from ..tool_parser import ToolParser
from ..types import CompletionMessage, ToolCall


class Param(BaseModel):
    name: str
    value: str | int | float | bool


class Action(BaseModel):
    tool_name: str
    tool_params: list[Param]


class ReActOutput(BaseModel):
    thought: str
    action: Action | None
    answer: str | None


class ReActToolParser(ToolParser):
    def get_tool_calls(self, output_message: CompletionMessage) -> list[ToolCall]:
        tool_calls = []
        response_text = str(output_message.content)
        try:
            react_output = ReActOutput.model_validate_json(response_text)
        except ValidationError as e:
            print(f"Error parsing action: {e}")
            return tool_calls

        if react_output.answer:
            return tool_calls

        if react_output.action:
            tool_name = react_output.action.tool_name
            tool_params = react_output.action.tool_params
            params = {param.name: param.value for param in tool_params}
            if tool_name and tool_params:
                call_id = str(uuid.uuid4())
                tool_calls = [
                    ToolCall(
                        call_id=call_id,
                        tool_name=tool_name,
                        arguments=json.dumps(params),
                    )
                ]

        return tool_calls
