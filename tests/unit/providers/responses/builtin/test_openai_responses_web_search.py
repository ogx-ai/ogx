# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import pytest
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk, Choice, ChoiceDelta

from ogx.core.datatypes import VectorStoresConfig, WebSearchAnnotationParams
from ogx.providers.inline.responses.builtin.responses.tool_executor import _web_search_citation_marker_id
from ogx_api.openai_responses import (
    OpenAIResponseInputToolWebSearch,
    OpenAIResponseMessage,
    OpenAIResponseObjectStreamResponseContentPartDone,
    OpenAIResponseObjectStreamResponseOutputTextDelta,
)
from ogx_api.responses.models import CreateResponseRequest
from ogx_api.tools import ToolDef, ToolInvocationResult
from tests.unit.providers.responses.builtin.test_openai_responses_helpers import fake_stream


@pytest.fixture(autouse=True)
def _register_web_search_tool(openai_responses_impl):
    """Every test in this file calls web_search, so register it for all of them rather than
    repeating the same ToolDef setup per test."""
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


async def test_web_search_output_includes_action_with_sources(openai_responses_impl, mock_inference_api):
    """Test that web search output includes action with source URLs."""
    input_text = "Find papers on transformers"
    model = "meta-llama/Llama-3.1-8B-Instruct"

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


async def test_malformed_source_does_not_fail_a_web_search_that_already_succeeded(
    openai_responses_impl, mock_inference_api
):
    """Regression test for a review finding: a non-dict source (or a non-string url/title)
    used to raise while building citation markers, from inside the same try/except as the
    tool invocation -- so a web_search call that already succeeded, with real results, got
    reported as a failed tool call. Citation-marker building must not be able to do that; the
    good source must still come through, and the call must still report completed.
    """
    openai_responses_impl.tool_runtime_api.invoke_tool.return_value = ToolInvocationResult(
        status="completed",
        content="Attention Is All You Need",
        metadata={
            "query": "Find papers on transformers",
            "sources": [
                "not a dict",
                {"title": "missing url"},
                {"url": "https://arxiv.org/abs/1706.03762", "title": "Attention Is All You Need"},
            ],
        },
    )
    mock_inference_api.openai_chat_completion.side_effect = [
        fake_stream("tool_call_completion.yaml"),
        fake_stream(),
    ]

    result = await openai_responses_impl.create_openai_response(
        CreateResponseRequest(
            input="Find papers on transformers",
            model="meta-llama/Llama-3.1-8B-Instruct",
            tools=[OpenAIResponseInputToolWebSearch(type="web_search")],
        )
    )

    web_search_output = next((o for o in result.output if o.type == "web_search_call"), None)
    assert web_search_output is not None
    assert web_search_output.status == "completed"
    assert web_search_output.action is not None
    assert len(web_search_output.action.sources) == 1
    assert web_search_output.action.sources[0].url == "https://arxiv.org/abs/1706.03762"


async def test_citation_annotations_can_be_disabled_via_config(openai_responses_impl, mock_inference_api):
    """Regression test for a review finding: the citation instruction was hard-coded with no
    off switch, unlike file_search's AnnotationPromptParams.enable_annotations.
    WebSearchAnnotationParams.enable_annotations=False must mean no citation instruction is
    sent at all (not just that annotations end up empty), end to end through ToolExecutor.
    """
    openai_responses_impl.tool_executor.vector_stores_config = VectorStoresConfig(
        web_search_annotation_params=WebSearchAnnotationParams(enable_annotations=False)
    )
    openai_responses_impl.tool_runtime_api.invoke_tool.return_value = ToolInvocationResult(
        status="completed",
        content="Attention Is All You Need",
        metadata={
            "query": "Find papers on transformers",
            "sources": [{"url": "https://arxiv.org/abs/1706.03762", "title": "Attention Is All You Need"}],
        },
    )
    mock_inference_api.openai_chat_completion.side_effect = [
        fake_stream("tool_call_completion.yaml"),
        fake_stream(),
    ]

    result = await openai_responses_impl.create_openai_response(
        CreateResponseRequest(
            input="Find papers on transformers",
            model="meta-llama/Llama-3.1-8B-Instruct",
            tools=[OpenAIResponseInputToolWebSearch(type="web_search")],
        )
    )

    # action.sources is unaffected -- the off switch only concerns the citation instruction.
    web_search_output = next((o for o in result.output if o.type == "web_search_call"), None)
    assert len(web_search_output.action.sources) == 1
    # The model's own second-round call never received a citation instruction.
    second_round_params = mock_inference_api.openai_chat_completion.call_args_list[1].args[0]
    tool_result_message = next(m for m in second_round_params.messages if getattr(m, "role", None) == "tool")
    assert "Cite a source" not in tool_result_message.content
    assert "Sources:" not in tool_result_message.content


def _setup_web_search_mocks(openai_responses_impl, mock_inference_api):
    """Set up common mocks for web search config tests."""
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
        CreateResponseRequest(
            input="What is the capital of Ireland?",
            model="meta-llama/Llama-3.1-8B-Instruct",
            temperature=0.1,
            tools=[OpenAIResponseInputToolWebSearch(type="web_search")],
        )
    )
    call_kwargs = openai_responses_impl.tool_runtime_api.invoke_tool.call_args.kwargs["kwargs"]
    assert call_kwargs["query"] == "What is the capital of Ireland?"
    assert "search_context_size" not in call_kwargs
    assert "allowed_domains" not in call_kwargs
    assert "user_location" not in call_kwargs


async def test_web_search_citation_populated_when_model_cites_inline(openai_responses_impl, mock_inference_api):
    """Regression test for #6763: web_search never produced url_citation annotations.

    Round 1: the model calls web_search. Round 2: the model answers, citing the
    retrieved source with the instructed `<|url-...|>` marker. The final response's
    output_text annotations should contain a matching url_citation.
    """
    openai_responses_impl.tool_runtime_api.invoke_tool.return_value = ToolInvocationResult(
        status="completed",
        content="Attention Is All You Need",
        metadata={
            "query": "What is a transformer?",
            "sources": [
                {"url": "https://arxiv.org/abs/1706.03762", "title": "Attention Is All You Need"},
            ],
        },
    )

    marker_id = _web_search_citation_marker_id("https://arxiv.org/abs/1706.03762")

    async def cited_final_answer():
        yield ChatCompletionChunk(
            id="chat-completion-web-cited",
            created=1234567893,
            model="meta-llama/Llama-3.1-8B-Instruct",
            object="chat.completion.chunk",
            choices=[
                Choice(
                    index=0,
                    delta=ChoiceDelta(
                        role="assistant",
                        content=f"Transformers use self-attention <|{marker_id}|>.",
                    ),
                    finish_reason="stop",
                )
            ],
        )

    mock_inference_api.openai_chat_completion.side_effect = [
        fake_stream("tool_call_completion.yaml"),
        cited_final_answer(),
    ]

    result = await openai_responses_impl.create_openai_response(
        CreateResponseRequest(
            input="What is a transformer?",
            model="meta-llama/Llama-3.1-8B-Instruct",
            tools=[OpenAIResponseInputToolWebSearch(type="web_search")],
        )
    )

    messages = [item for item in result.output if isinstance(item, OpenAIResponseMessage)]
    assert len(messages) == 1
    text_content = messages[0].content[0]
    assert text_content.text == "Transformers use self-attention."
    assert len(text_content.annotations) == 1
    annotation = text_content.annotations[0]
    assert annotation.type == "url_citation"
    assert annotation.url == "https://arxiv.org/abs/1706.03762"
    assert annotation.title == "Attention Is All You Need"


async def test_web_search_citation_falls_back_when_model_does_not_cite(openai_responses_impl, mock_inference_api):
    """When the model doesn't echo the `<|url-...|>` marker, the response should still
    attribute the answer to the single retrieved source rather than returning no
    annotations at all."""
    openai_responses_impl.tool_runtime_api.invoke_tool.return_value = ToolInvocationResult(
        status="completed",
        content="Attention Is All You Need",
        metadata={
            "query": "What is a transformer?",
            "sources": [
                {"url": "https://arxiv.org/abs/1706.03762", "title": "Attention Is All You Need"},
            ],
        },
    )

    async def uncited_final_answer():
        yield ChatCompletionChunk(
            id="chat-completion-web-uncited",
            created=1234567894,
            model="meta-llama/Llama-3.1-8B-Instruct",
            object="chat.completion.chunk",
            choices=[
                Choice(
                    index=0,
                    delta=ChoiceDelta(
                        role="assistant",
                        content="Transformers use self-attention.",
                    ),
                    finish_reason="stop",
                )
            ],
        )

    mock_inference_api.openai_chat_completion.side_effect = [
        fake_stream("tool_call_completion.yaml"),
        uncited_final_answer(),
    ]

    result = await openai_responses_impl.create_openai_response(
        CreateResponseRequest(
            input="What is a transformer?",
            model="meta-llama/Llama-3.1-8B-Instruct",
            tools=[OpenAIResponseInputToolWebSearch(type="web_search")],
        )
    )

    messages = [item for item in result.output if isinstance(item, OpenAIResponseMessage)]
    assert len(messages) == 1
    text_content = messages[0].content[0]
    assert text_content.text == "Transformers use self-attention."
    assert len(text_content.annotations) == 1
    assert text_content.annotations[0].url == "https://arxiv.org/abs/1706.03762"


@pytest.mark.parametrize(
    "split_marker",
    [
        pytest.param(
            lambda marker_id: ["<", "|", "url", "-", marker_id.split("-", 1)[1], "|", ">."], id="char_by_char"
        ),
        pytest.param(lambda marker_id: ["<|u", f"rl-{marker_id.split('-', 1)[1]}|>."], id="split_mid_prefix"),
        pytest.param(lambda marker_id: [f"<|{marker_id}|", ">."], id="split_before_close"),
        pytest.param(lambda marker_id: ["[", f"{marker_id}]."], id="bracket_form"),
    ],
)
async def test_streaming_deltas_never_leak_partial_url_marker_syntax(
    openai_responses_impl, mock_inference_api, split_marker
):
    """Regression test: a `url-...` marker must be withheld from streamed deltas at every
    intermediate boundary a real provider might split it at, not just once the literal
    "url-" word has fully arrived. Before the fix, "<", "|", "url" (no trailing "-") would
    already be flushed as visible text, since the pending-marker check only recognized a
    partial "file-" prefix, not a partial "url-" one.
    """
    openai_responses_impl.tool_runtime_api.invoke_tool.return_value = ToolInvocationResult(
        status="completed",
        content="Attention Is All You Need",
        metadata={
            "query": "What is a transformer?",
            "sources": [{"url": "https://arxiv.org/abs/1706.03762", "title": "Attention Is All You Need"}],
        },
    )
    marker_id = _web_search_citation_marker_id("https://arxiv.org/abs/1706.03762")
    chunks_content = ["Transformers use self-attention ", *split_marker(marker_id)]

    async def split_marker_final_answer():
        for i, delta_text in enumerate(chunks_content):
            yield ChatCompletionChunk(
                id="chat-completion-web-split",
                created=1234567895,
                model="meta-llama/Llama-3.1-8B-Instruct",
                object="chat.completion.chunk",
                choices=[
                    Choice(
                        index=0,
                        delta=ChoiceDelta(role="assistant", content=delta_text),
                        finish_reason="stop" if i == len(chunks_content) - 1 else None,
                    )
                ],
            )

    mock_inference_api.openai_chat_completion.side_effect = [
        fake_stream("tool_call_completion.yaml"),
        split_marker_final_answer(),
    ]

    stream = await openai_responses_impl.create_openai_response(
        CreateResponseRequest(
            input="What is a transformer?",
            model="meta-llama/Llama-3.1-8B-Instruct",
            tools=[OpenAIResponseInputToolWebSearch(type="web_search")],
            stream=True,
        )
    )

    delta_texts = []
    content_part_done_text = None
    async for event in stream:
        if isinstance(event, OpenAIResponseObjectStreamResponseOutputTextDelta):
            delta_texts.append(event.delta)
            assert "<|url" not in event.delta, "raw marker syntax must never leak into a delta"
            assert "[url-" not in event.delta, "raw marker syntax must never leak into a delta"
            assert "|>" not in event.delta, "raw marker syntax must never leak into a delta"
        elif isinstance(event, OpenAIResponseObjectStreamResponseContentPartDone) and event.part.text:
            content_part_done_text = event.part.text

    assert content_part_done_text == "Transformers use self-attention."
    assert "".join(delta_texts) == content_part_done_text
