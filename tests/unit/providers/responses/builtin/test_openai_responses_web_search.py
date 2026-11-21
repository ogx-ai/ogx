# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import pytest

from ogx.web_search import WebSearchResult
from ogx_api import InvalidParameterError, WebSearchSource
from ogx_api.openai_responses import (
    OpenAIResponseInputToolWebSearch,
    WebSearchToolTypes,
)
from ogx_api.responses.models import CreateResponseRequest
from ogx_api.router_utils import try_translate_to_http_exception
from ogx_api.tools import ToolDef
from tests.unit.providers.responses.builtin.test_openai_responses_helpers import fake_stream


async def test_web_search_output_includes_action_with_sources(openai_responses_impl, mock_inference_api):
    """Test that web search output includes action with source URLs."""
    input_text = "Find papers on transformers"
    model = "meta-llama/Llama-3.1-8B-Instruct"

    openai_responses_impl.web_search_api.tool_def.return_value = ToolDef(
        name="web_search",
        toolgroup_id="web_search",
        description="Search the web for information",
        input_schema={
            "type": "object",
            "properties": {"query": {"type": "string", "description": "The query to search for"}},
            "required": ["query"],
        },
    )

    openai_responses_impl.web_search_api.search.return_value = WebSearchResult(
        content="Attention Is All You Need",
        query="What is the capital of Ireland?",
        sources=[
            WebSearchSource(url="https://arxiv.org/abs/1706.03762"),
            WebSearchSource(url="https://en.wikipedia.org/wiki/Transformer"),
        ],
    )

    mock_inference_api.openai_chat_completion.side_effect = [
        fake_stream("tool_call_completion.yaml"),
        fake_stream(),
    ]

    result = await openai_responses_impl.create_openai_response(
        CreateResponseRequest(
            input=input_text, model=model, temperature=0.1, tools=[OpenAIResponseInputToolWebSearch(type="web_search")]
        )
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
    openai_responses_impl.web_search_api.tool_def.return_value = ToolDef(
        name="web_search",
        toolgroup_id="web_search",
        description="Search the web",
        input_schema={"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
    )
    openai_responses_impl.web_search_api.search.return_value = WebSearchResult(content="Dublin")
    mock_inference_api.openai_chat_completion.side_effect = [
        fake_stream("tool_call_completion.yaml"),
        fake_stream(),
    ]


async def test_web_search_with_filters_and_location(openai_responses_impl, mock_inference_api):
    """Test that web search filters and user_location are passed to the web search API."""
    _setup_web_search_mocks(openai_responses_impl, mock_inference_api)
    await openai_responses_impl.create_openai_response(
        CreateResponseRequest(
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
    )
    call = openai_responses_impl.web_search_api.search.call_args
    assert call.args[1] == "What is the capital of Ireland?"
    assert call.kwargs["allowed_domains"] == ["arxiv.org", "scholar.google.com"]
    assert call.kwargs["user_location"]["country"] == "US"
    assert call.kwargs["user_location"]["city"] == "San Francisco"
    assert call.kwargs["search_context_size"] == "high"


async def test_web_search_without_config_passes_only_query(openai_responses_impl, mock_inference_api):
    """Test that web search without filters/location only passes query."""
    _setup_web_search_mocks(openai_responses_impl, mock_inference_api)
    await openai_responses_impl.create_openai_response(
        CreateResponseRequest(
            input="What is the capital of Ireland?",
            model="meta-llama/Llama-3.1-8B-Instruct",
            temperature=0.1,
            tools=[OpenAIResponseInputToolWebSearch(type="web_search")],
        )
    )
    call = openai_responses_impl.web_search_api.search.call_args
    assert call.args[1] == "What is the capital of Ireland?"
    assert call.kwargs == {}


@pytest.mark.parametrize("stream", [True, False], ids=["streaming", "non_streaming"])
@pytest.mark.parametrize("tool_type", WebSearchToolTypes)
async def test_unregistered_web_search_tool_is_rejected_with_400_before_the_response_starts(
    openai_responses_impl, mock_inference_api, tool_type, stream
):
    openai_responses_impl.web_search_api = None

    with pytest.raises(InvalidParameterError, match="web_search") as exc_info:
        await openai_responses_impl.create_openai_response(
            CreateResponseRequest(
                input="What is the capital of Ireland?",
                model="meta-llama/Llama-3.1-8B-Instruct",
                stream=stream,
                store=False,
                tools=[OpenAIResponseInputToolWebSearch(type=tool_type)],
            )
        )

    http_exc = try_translate_to_http_exception(exc_info.value)
    assert http_exc is not None
    assert http_exc.status_code == 400
    mock_inference_api.openai_chat_completion.assert_not_called()


@pytest.mark.parametrize("stream", [True, False], ids=["streaming", "non_streaming"])
@pytest.mark.parametrize("tool_type", WebSearchToolTypes)
async def test_empty_web_search_router_is_rejected_with_400_before_the_response_starts(
    openai_responses_impl, mock_inference_api, tool_type, stream
):
    # The web_search auto-router is wired in-process even when no web_search
    # providers are configured, so the API is an empty router rather than None.
    # The request must fail pre-stream instead of mid-stream when tool_def(None)
    # is fetched after response.created has been sent.
    openai_responses_impl.web_search_api.available_providers.return_value = []

    with pytest.raises(InvalidParameterError, match="web_search") as exc_info:
        await openai_responses_impl.create_openai_response(
            CreateResponseRequest(
                input="What is the capital of Ireland?",
                model="meta-llama/Llama-3.1-8B-Instruct",
                stream=stream,
                store=False,
                tools=[OpenAIResponseInputToolWebSearch(type=tool_type)],
            )
        )

    http_exc = try_translate_to_http_exception(exc_info.value)
    assert http_exc is not None
    assert http_exc.status_code == 400
    mock_inference_api.openai_chat_completion.assert_not_called()
