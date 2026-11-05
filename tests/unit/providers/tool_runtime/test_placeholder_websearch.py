# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import pytest

from ogx.providers.inline.tool_runtime.placeholder.config import PlaceholderWebSearchToolConfig
from ogx.providers.inline.tool_runtime.placeholder.placeholder import PlaceholderWebSearchToolRuntimeImpl


@pytest.fixture
def placeholder():
    return PlaceholderWebSearchToolRuntimeImpl(PlaceholderWebSearchToolConfig())


async def test_sample_run_config_is_empty():
    assert PlaceholderWebSearchToolConfig.sample_run_config(__distro_dir__="/tmp") == {}


async def test_list_runtime_tools_returns_web_search(placeholder):
    response = await placeholder.list_runtime_tools()
    assert len(response.data) == 1
    tool = response.data[0]
    assert tool.name == "web_search"
    assert tool.input_schema["required"] == ["query"]
    assert "not configured" in tool.description


async def test_invoke_returns_not_available_notice(placeholder):
    result = await placeholder.invoke_tool("web_search", {"query": "capital of France"})
    assert "Web search is not available on this server" in result.content
    assert "Configure a web search provider" in result.content
    assert result.metadata["query"] == "capital of France"
    assert result.metadata["sources"] == []
