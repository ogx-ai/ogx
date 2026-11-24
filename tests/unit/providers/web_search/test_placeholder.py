# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import pytest

from ogx.providers.inline.web_search.placeholder.config import PlaceholderWebSearchConfig
from ogx.providers.inline.web_search.placeholder.placeholder import PlaceholderWebSearchImpl


@pytest.fixture
def placeholder():
    return PlaceholderWebSearchImpl(PlaceholderWebSearchConfig())


async def test_sample_run_config_is_empty():
    assert PlaceholderWebSearchConfig.sample_run_config(__distro_dir__="/tmp") == {}


async def test_tool_def_returns_web_search(placeholder):
    tool = await placeholder.tool_def()
    assert tool.name == "web_search"
    assert tool.input_schema["required"] == ["query"]
    assert "not configured" in tool.description


async def test_search_returns_not_available_notice(placeholder):
    result = await placeholder.search(None, "capital of France")
    assert "Web search is not available on this server" in result.content
    assert "Configure a web search provider" in result.content
    assert result.query == "capital of France"
    assert result.sources == []
