# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from unittest.mock import AsyncMock, patch

import pytest

from ogx.providers.remote.inference.anthropic.anthropic import AnthropicInferenceAdapter
from ogx.providers.remote.inference.anthropic.config import AnthropicConfig
from ogx_api.inference.models import (
    OpenAIChatCompletionRequestWithExtraBody,
    OpenAIResponseFormatJSONSchema,
    OpenAIUserMessageParam,
)

MIXIN_PATH = "ogx.providers.utils.inference.openai_mixin.OpenAIMixin.openai_chat_completion"


@pytest.fixture
def adapter():
    config = AnthropicConfig(api_key="test-key")
    return AnthropicInferenceAdapter(config=config)


@pytest.mark.parametrize(
    "input_params,expected_params",
    [
        ({}, {"type": "object"}),
        ({"type": "object", "properties": {}}, {"type": "object", "properties": {}}),
    ],
    ids=["empty", "already-valid"],
)
async def test_empty_tool_parameters_normalized(adapter, input_params, expected_params):
    """Anthropic rejects parameters: {} but OpenAI accepts it; the adapter normalizes."""
    params = OpenAIChatCompletionRequestWithExtraBody(
        model="claude-sonnet-4-6",
        messages=[{"role": "user", "content": "hi"}],
        tools=[{"type": "function", "function": {"name": "my_func", "parameters": input_params}}],
    )

    with patch.object(type(adapter).__mro__[1], "openai_chat_completion", new_callable=AsyncMock) as mock_super:
        mock_super.return_value = {}
        await adapter.openai_chat_completion(params)

    assert params.tools[0]["function"]["parameters"] == expected_params


def _request(strict=...):
    json_schema = {"name": "s", "schema": {"type": "object"}}
    if strict is not ...:
        json_schema["strict"] = strict
    return OpenAIChatCompletionRequestWithExtraBody(
        model="claude-sonnet-4-6",
        messages=[OpenAIUserMessageParam(role="user", content="hi")],
        response_format=OpenAIResponseFormatJSONSchema(json_schema=json_schema),
    )


@pytest.mark.parametrize(
    "given, expected",
    [(..., False), (None, False), (True, True), (False, False)],
)
async def test_json_schema_strict_defaulted_for_anthropic(given, expected):
    adapter = AnthropicInferenceAdapter(config=AnthropicConfig(api_key="test"))
    params = _request(strict=given)

    with patch(MIXIN_PATH, new=AsyncMock(return_value=None)) as mock_super:
        await adapter.openai_chat_completion(params)

    forwarded = mock_super.call_args.args[0]
    assert forwarded.response_format.json_schema["strict"] is expected
