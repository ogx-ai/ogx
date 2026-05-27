# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from ogx_api.openai_responses import (
    OpenAIResponseInputToolWebSearch,
)
from ogx_api.tools import ToolDef, ToolInvocationResult
from tests.unit.providers.responses.builtin.test_openai_responses_helpers import fake_stream


async def test_web_search_output_includes_action_with_sources(openai_responses_impl, mock_inference_api):
    """Test that web search output includes action with source URLs."""
    input_text = "Find papers on transformers"
    model = "meta-llama/Llama-3.1-8B-Instruct"

    openai_responses_impl.tool_groups_api.get_tool.return_value = ToolDef(
        name="web_search",
        toolgroup_id="web_search",
        description="Search the web for information",
        input_schema={
            "type": "object",
            "properties": {"query": {"type": "string", "description": "The query to search for"}},
            "required": ["query"],
        },
    )

    openai_responses_impl.tool_runtime_api.invoke_tool.return_value = ToolInvocationResult(
        status="completed",
        content="Attention Is All You Need",
        metadata={
            "query": "What is the capital of Ireland?",
            "sources": [
                {"url": "https://arxiv.org/abs/1706.03762"},
                {"url": "https://en.wikipedia.org/wiki/Transformer"},
            ],
        },
    )

    mock_inference_api.openai_chat_completion.side_effect = [
        fake_stream("tool_call_completion.yaml"),
        fake_stream(),
    ]

    result = await openai_responses_impl.create_openai_response(
        input=input_text,
        model=model,
        temperature=0.1,
        tools=[OpenAIResponseInputToolWebSearch(type="web_search")],
    )

    # Find the web search tool call in output
    web_search_output = next(
        (o for o in result.output if o.type == "web_search_call"),
        None,
    )
    assert web_search_output is not None
    assert web_search_output.action is not None
    assert web_search_output.action.type == "search"
    assert len(web_search_output.action.sources) == 2
    assert web_search_output.action.sources[0].url == "https://arxiv.org/abs/1706.03762"


def _setup_web_search_mocks(openai_responses_impl, mock_inference_api):
    """Set up common mocks for web search config tests."""
    openai_responses_impl.tool_groups_api.get_tool.return_value = ToolDef(
        name="web_search",
        toolgroup_id="web_search",
        description="Search the web",
        input_schema={"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
    )
    openai_responses_impl.tool_runtime_api.invoke_tool.return_value = ToolInvocationResult(
        status="completed",
        content="Dublin",
    )
    mock_inference_api.openai_chat_completion.side_effect = [
        fake_stream("tool_call_completion.yaml"),
        fake_stream(),
    ]


async def test_web_search_with_filters_and_location(openai_responses_impl, mock_inference_api):
    """Test that web search filters and user_location are passed to invoke_tool."""
    _setup_web_search_mocks(openai_responses_impl, mock_inference_api)
    await openai_responses_impl.create_openai_response(
        input="What is the capital of Ireland?",
        model="meta-llama/Llama-3.1-8B-Instruct",
        temperature=0.1,
        tools=[
            OpenAIResponseInputToolWebSearch(
                type="web_search",
                search_context_size="high",
                filters={"allowed_domains": ["arxiv.org", "scholar.google.com"]},
                user_location={"type": "approximate", "country": "US", "city": "San Francisco"},
            )
        ],
    )
    call_kwargs = openai_responses_impl.tool_runtime_api.invoke_tool.call_args.kwargs["kwargs"]
    assert call_kwargs["query"] == "What is the capital of Ireland?"
    assert call_kwargs["allowed_domains"] == ["arxiv.org", "scholar.google.com"]
    assert call_kwargs["user_location"]["country"] == "US"
    assert call_kwargs["user_location"]["city"] == "San Francisco"
    assert call_kwargs["search_context_size"] == "high"


async def test_web_search_without_config_passes_only_query(openai_responses_impl, mock_inference_api):
    """Test that web search without filters/location only passes query."""
    _setup_web_search_mocks(openai_responses_impl, mock_inference_api)
    await openai_responses_impl.create_openai_response(
        input="What is the capital of Ireland?",
        model="meta-llama/Llama-3.1-8B-Instruct",
        temperature=0.1,
        tools=[OpenAIResponseInputToolWebSearch(type="web_search")],
    )
    call_kwargs = openai_responses_impl.tool_runtime_api.invoke_tool.call_args.kwargs["kwargs"]
    assert call_kwargs["query"] == "What is the capital of Ireland?"
    assert "search_context_size" not in call_kwargs
    assert "allowed_domains" not in call_kwargs
    assert "user_location" not in call_kwargs
