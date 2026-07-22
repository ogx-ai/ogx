# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import pytest

from ogx.providers.remote.inference.anthropic.anthropic import AnthropicInferenceAdapter, _make_schema_strict
from ogx.providers.remote.inference.anthropic.config import AnthropicConfig
from ogx_api import Model, OpenAIChatCompletionRequestWithExtraBody, OpenAIUserMessageParam
from ogx_api.inference.models import OpenAIJSONSchema, OpenAIResponseFormatJSONSchema


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


async def _empty_stream():
    if False:
        yield None


async def test_chat_completion_defaults_strict_true_when_none():
    adapter = AnthropicInferenceAdapter(config=AnthropicConfig(api_key="test-key"))
    adapter.__provider_id__ = "anthropic"
    adapter.model_store = AsyncMock()
    adapter.model_store.get_model.return_value = Model(
        identifier="test-model",
        provider_id="anthropic",
        provider_resource_id="test-model",
    )

    mock_client = MagicMock()
    captured_params = {}

    async def _capture_create(**kwargs):
        captured_params.update(kwargs)
        return _empty_stream()

    mock_client.chat.completions.create = _capture_create

    with patch.object(type(adapter), "client", new_callable=PropertyMock, return_value=mock_client):
        params = OpenAIChatCompletionRequestWithExtraBody(
            model="test-model",
            messages=[OpenAIUserMessageParam(role="user", content="test")],
            response_format=OpenAIResponseFormatJSONSchema(
                json_schema=OpenAIJSONSchema(
                    name="test",
                    schema={"type": "object", "properties": {"a": {"type": "string"}}},
                ),
            ),
        )

        await adapter.openai_chat_completion(params)

    assert captured_params["response_format"]["json_schema"]["strict"] is True
    assert captured_params["response_format"]["json_schema"]["schema"]["additionalProperties"] is False


def test_make_schema_strict_adds_additional_properties():
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
        },
    }
    _make_schema_strict(schema)
    assert schema["additionalProperties"] is False
    assert "required" not in schema


def test_make_schema_strict_preserves_existing():
    schema = {
        "type": "object",
        "properties": {"a": {"type": "string"}},
        "additionalProperties": True,
        "required": ["a"],
    }
    _make_schema_strict(schema)
    assert schema["additionalProperties"] is True
    assert schema["required"] == ["a"]
