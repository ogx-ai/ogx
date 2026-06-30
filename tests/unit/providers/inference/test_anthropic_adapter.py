# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from unittest.mock import AsyncMock, patch

import pytest

from ogx.providers.remote.inference.anthropic.anthropic import AnthropicInferenceAdapter
from ogx.providers.remote.inference.anthropic.config import AnthropicConfig
from ogx_api.inference.models import OpenAIChatCompletionRequestWithExtraBody


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
