# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Integration tests for built-in tool providers via the Responses API.

Built-in tools (web_search, etc.) are auto-registered internally and accessed
through the Responses API tool definitions.
"""

import os

import pytest


def test_web_search_tool(ogx_client, text_model_id):
    """Test the web search tool functionality via the Responses API."""
    if "TAVILY_SEARCH_API_KEY" not in os.environ:
        pytest.skip("TAVILY_SEARCH_API_KEY not set, skipping test")

    response = ogx_client.responses.create(
        model=text_model_id,
        input="What are the latest developments in quantum computing?",
        tools=[{"type": "web_search"}],
        stream=False,
    )

    assert response is not None
    assert response.status == "completed"

    web_search_calls = [item for item in response.output if item.type == "web_search_call"]
    assert web_search_calls, "Expected at least one web_search_call in the response output"
