# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any
from unittest.mock import AsyncMock, patch

from ogx.providers.remote.inference.fireworks.config import FireworksImplConfig
from ogx.providers.remote.inference.fireworks.fireworks import FireworksInferenceAdapter
from ogx.providers.utils.inference.openai_mixin import OpenAIMixin
from ogx_api import OpenAIChatCompletionRequestWithExtraBody, OpenAICompletionRequestWithExtraBody


def _make_adapter() -> FireworksInferenceAdapter:
    return FireworksInferenceAdapter(config=FireworksImplConfig(api_key="test-key"))


def _make_params(tools: list[dict[str, Any]] | None) -> OpenAIChatCompletionRequestWithExtraBody:
    return OpenAIChatCompletionRequestWithExtraBody(
        model="accounts/fireworks/models/gpt-oss-120b",
        messages=[{"role": "user", "content": "test"}],
        tools=tools,
    )


async def test_strips_type_from_function_tool_definitions():
    adapter = _make_adapter()
    tools = [
        {
            "type": "function",
            "function": {
                "type": "function",
                "name": "get_current_time",
                "description": "Get the current time",
                "parameters": {},
            },
        }
    ]
    params = _make_params(tools)

    with patch.object(OpenAIMixin, "openai_chat_completion", new=AsyncMock(return_value="completed")) as mock_super:
        await adapter.openai_chat_completion(params=params)

    forwarded = mock_super.call_args.args[0]
    assert forwarded.tools[0]["function"] == {
        "name": "get_current_time",
        "description": "Get the current time",
        "parameters": {},
    }
    assert "type" in params.tools[0]["function"]


async def test_tools_without_type_are_forwarded_unchanged():
    adapter = _make_adapter()
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_current_time",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]
    params = _make_params(tools)

    with patch.object(OpenAIMixin, "openai_chat_completion", new=AsyncMock(return_value="completed")) as mock_super:
        await adapter.openai_chat_completion(params=params)

    forwarded = mock_super.call_args.args[0]
    assert forwarded.tools == tools


async def test_no_tools_leaves_params_unchanged():
    adapter = _make_adapter()
    params = _make_params(None)

    with patch.object(OpenAIMixin, "openai_chat_completion", new=AsyncMock(return_value="completed")) as mock_super:
        await adapter.openai_chat_completion(params=params)

    assert mock_super.call_args.args[0] is params


def _make_completion_params(
    stream: bool, stream_options: dict[str, Any] | None
) -> OpenAICompletionRequestWithExtraBody:
    return OpenAICompletionRequestWithExtraBody(
        model="accounts/fireworks/models/gpt-oss-120b",
        prompt="test",
        stream=stream,
        stream_options=stream_options,
    )


async def test_completion_stream_opts_out_of_unrequested_usage():
    adapter = _make_adapter()
    params = _make_completion_params(stream=True, stream_options=None)

    with patch.object(OpenAIMixin, "openai_completion", new=AsyncMock(return_value="stream")) as mock_super:
        result = await adapter.openai_completion(params=params)

    assert result == "stream"
    forwarded = mock_super.call_args.args[0]
    assert forwarded is not params
    assert forwarded.stream_options == {"include_usage": False}


async def test_completion_stream_merges_opt_out_with_existing_stream_options():
    adapter = _make_adapter()
    params = _make_completion_params(stream=True, stream_options={"include_obfuscation": False})

    with patch.object(OpenAIMixin, "openai_completion", new=AsyncMock(return_value="stream")) as mock_super:
        await adapter.openai_completion(params=params)

    forwarded = mock_super.call_args.args[0]
    assert forwarded.stream_options == {"include_obfuscation": False, "include_usage": False}


async def test_completion_stream_preserves_requested_usage():
    adapter = _make_adapter()
    params = _make_completion_params(stream=True, stream_options={"include_usage": True})

    with patch.object(OpenAIMixin, "openai_completion", new=AsyncMock(return_value="stream")) as mock_super:
        result = await adapter.openai_completion(params=params)

    assert result == "stream"
    assert mock_super.call_args.args[0] is params


async def test_completion_non_stream_leaves_stream_options_untouched():
    adapter = _make_adapter()
    params = _make_completion_params(stream=False, stream_options=None)

    with patch.object(OpenAIMixin, "openai_completion", new=AsyncMock(return_value="response")) as mock_super:
        result = await adapter.openai_completion(params=params)

    assert result == "response"
    assert mock_super.call_args.args[0] is params
