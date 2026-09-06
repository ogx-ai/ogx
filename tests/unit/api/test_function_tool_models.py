# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import pytest
from pydantic import ValidationError

from ogx_api.openai_responses import OpenAIResponseInputToolFunction


def test_function_tool_with_valid_parameters():
    tool = OpenAIResponseInputToolFunction(
        name="get_weather",
        parameters={"type": "object", "properties": {"city": {"type": "string"}}},
    )
    assert tool.parameters is not None
    assert tool.parameters["type"] == "object"


def test_function_tool_with_none_parameters():
    tool = OpenAIResponseInputToolFunction(name="get_time", parameters=None)
    assert tool.parameters is None


def test_function_tool_with_empty_parameters():
    tool = OpenAIResponseInputToolFunction(name="get_time", parameters={})
    assert tool.parameters == {}


def test_function_tool_parameters_missing_type_rejected():
    with pytest.raises(ValidationError, match="must include a top-level 'type' field"):
        OpenAIResponseInputToolFunction(
            name="get_time",
            parameters={"properties": {"tz": {"type": "string"}}},
        )
