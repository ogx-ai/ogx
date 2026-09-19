# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest
from pydantic import ValidationError

from ogx.core.datatypes import VectorStoresConfig
from ogx.core.exceptions import translate_exception
from ogx.providers.inline.responses.builtin.responses.tool_executor import ToolExecutor
from ogx.providers.utils.responses.responses_store import _OpenAIResponseObjectWithInputAndMessages
from ogx_api.common.errors import InvalidParameterError
from ogx_api.openai_responses import OpenAIResponseInputToolFileSearch
from ogx_api.responses.fastapi_routes import sse_generator
from ogx_api.responses.models import CreateResponseRequest
from ogx_api.vector_io import SearchRankingOptions, VectorStoreSearchResponsePage
from tests.unit.providers.responses.builtin.test_openai_responses_helpers import fake_stream


def test_ranking_options_schema_documents_supported_rankers():
    """The API schema should explain which ranker values OGX supports."""
    ranker_schema = SearchRankingOptions.model_json_schema()["properties"]["ranker"]

    description = ranker_schema.get("description")
    assert description
    for ranker in ("weighted", "rrf", "neural", "classifier"):
        assert ranker in description


async def test_file_search_forwards_ranking_options_weights(mock_vector_io_api):
    """Test that file_search forwards ranking_options.weights to vector store search."""
    query = "What is machine learning?"
    vector_store_id = "test_vector_store"
    ranking_options = SearchRankingOptions(
        ranker="rrf",
        weights={"vector": 1.0, "keyword": 0.0},
    )

    mock_vector_io_api.openai_search_vector_store.return_value = VectorStoreSearchResponsePage(
        search_query=[query],
        has_more=False,
        data=[],
    )
    tool_executor = ToolExecutor(
        tool_groups_api=None,  # type: ignore
        tool_runtime_api=None,  # type: ignore
        vector_io_api=mock_vector_io_api,
        vector_stores_config=VectorStoresConfig(),
        mcp_session_manager=None,
    )

    file_search_tool = OpenAIResponseInputToolFileSearch(
        vector_store_ids=[vector_store_id],
        ranking_options=ranking_options,
    )
    await tool_executor._execute_file_search_via_vector_store(
        query=query,
        response_file_search_tool=file_search_tool,
    )

    call_kwargs = mock_vector_io_api.openai_search_vector_store.call_args
    request = call_kwargs.kwargs["request"]
    assert request.ranking_options == ranking_options
    assert request.ranking_options.weights == {"vector": 1.0, "keyword": 0.0}


async def test_file_search_forwards_hybrid_search(mock_vector_io_api):
    """Test that file_search forwards hybrid_search and leaves the search mode choice to the vector store."""
    query = "What is machine learning?"
    vector_stores_config = VectorStoresConfig()
    assert vector_stores_config.chunk_retrieval_params.default_search_mode == "vector"

    mock_vector_io_api.openai_search_vector_store.return_value = VectorStoreSearchResponsePage(
        search_query=[query],
        has_more=False,
        data=[],
    )
    tool_executor = ToolExecutor(
        tool_groups_api=None,  # type: ignore
        tool_runtime_api=None,  # type: ignore
        vector_io_api=mock_vector_io_api,
        vector_stores_config=vector_stores_config,
        mcp_session_manager=None,
    )

    file_search_tool = OpenAIResponseInputToolFileSearch.model_validate(
        {
            "type": "file_search",
            "vector_store_ids": ["test_vector_store"],
            "ranking_options": {"ranker": "auto", "hybrid_search": {"embedding_weight": 1, "text_weight": 3}},
        }
    )
    await tool_executor._execute_file_search_via_vector_store(
        query=query,
        response_file_search_tool=file_search_tool,
    )

    request = mock_vector_io_api.openai_search_vector_store.call_args.kwargs["request"]
    assert request.search_mode == "vector"
    assert request.ranking_options.hybrid_search.model_dump() == {"embedding_weight": 1.0, "text_weight": 3.0}


@pytest.mark.parametrize("ranker", [None, "auto", "default-2024-11-15", "rrf"])
def test_file_search_tool_keeps_hybrid_search(ranker):
    """Test that hybrid_search sent by OpenAI clients is parsed and echoed back instead of being dropped."""
    file_search_tool = OpenAIResponseInputToolFileSearch.model_validate(
        {
            "type": "file_search",
            "vector_store_ids": ["test_vector_store"],
            "ranking_options": {"ranker": ranker, "hybrid_search": {"embedding_weight": 0.7, "text_weight": 0.3}},
        }
    )

    ranking_options = file_search_tool.model_dump()["ranking_options"]
    assert ranking_options["hybrid_search"] == {"embedding_weight": 0.7, "text_weight": 0.3}


@pytest.mark.parametrize(
    "ranking_options, error",
    [
        ({"hybrid_search": {"embedding_weight": 0, "text_weight": 0}}, "must not both be 0"),
        ({"hybrid_search": {"embedding_weight": -1, "text_weight": 1}}, "greater than or equal to 0"),
        ({"hybrid_search": {"embedding_weight": 1}}, "text_weight"),
        ({"hybrid_search": {"embedding_weight": float("inf"), "text_weight": 1}}, "finite number"),
        ({"hybrid_search": {"embedding_weight": 1e308, "text_weight": 1e308}}, "must have a finite sum"),
        (
            {"hybrid_search": {"embedding_weight": 1, "text_weight": 1}, "weights": {"vector": 0.5, "keyword": 0.5}},
            "hybrid_search cannot be combined with weights",
        ),
        (
            {"ranker": "weighted", "hybrid_search": {"embedding_weight": 1, "text_weight": 1}},
            "hybrid_search cannot be combined with ranker 'weighted'",
        ),
        (
            {"ranker": "neural", "model": "reranker", "hybrid_search": {"embedding_weight": 1, "text_weight": 1}},
            "hybrid_search cannot be combined with ranker 'neural'",
        ),
        (
            {"ranker": "classifier", "model": "classifier", "hybrid_search": {"embedding_weight": 1, "text_weight": 1}},
            "hybrid_search cannot be combined with ranker 'classifier'",
        ),
        (
            {"ranker": "normalized", "hybrid_search": {"embedding_weight": 1, "text_weight": 1}},
            "hybrid_search cannot be combined with ranker 'normalized'",
        ),
    ],
)
def test_ranking_options_reject_invalid_hybrid_search(ranking_options, error):
    """Test that invalid weights and options that hybrid_search would silently override are rejected."""
    with pytest.raises(ValidationError, match=error):
        SearchRankingOptions.model_validate(ranking_options)


def _unsupported_hybrid_search_message(vector_store_id: str = "test_vector_store") -> str:
    return f"The provider of vector store '{vector_store_id}' does not support weighted hybrid search."


def _unsupported_hybrid_search_error(vector_store_id: str = "test_vector_store") -> InvalidParameterError:
    """The error openai_search_vector_store raises for a store that cannot apply the weights."""
    return InvalidParameterError(
        "ranking_options.hybrid_search",
        {"embedding_weight": 0.3, "text_weight": 0.7},
        _unsupported_hybrid_search_message(vector_store_id),
    )


def _hybrid_search_file_search_tool() -> OpenAIResponseInputToolFileSearch:
    return OpenAIResponseInputToolFileSearch.model_validate(
        {
            "type": "file_search",
            "vector_store_ids": ["test_vector_store"],
            "ranking_options": {"hybrid_search": {"embedding_weight": 0.3, "text_weight": 0.7}},
        }
    )


def _tool_executor(mock_vector_io_api) -> ToolExecutor:
    return ToolExecutor(
        tool_groups_api=None,  # type: ignore
        tool_runtime_api=None,  # type: ignore
        vector_io_api=mock_vector_io_api,
        vector_stores_config=VectorStoresConfig(),
        mcp_session_manager=None,
    )


async def test_file_search_surfaces_unsupported_hybrid_search_error(mock_vector_io_api):
    """Test that file_search lets the 400 for an unsupported hybrid_search reach its caller.

    Returning no chunks here would leave the client with an HTTP 200 whose answer was written
    without any retrieval, and no way to tell that from a store that simply held nothing.
    """
    mock_vector_io_api.openai_search_vector_store.side_effect = _unsupported_hybrid_search_error()

    with pytest.raises(InvalidParameterError) as excinfo:
        await _tool_executor(mock_vector_io_api)._execute_file_search_via_vector_store(
            query="What is the travel reimbursement limit?",
            response_file_search_tool=_hybrid_search_file_search_tool(),
        )

    assert excinfo.value.status_code == httpx.codes.BAD_REQUEST
    assert "ranking_options.hybrid_search" in str(excinfo.value)


async def test_file_search_rejects_hybrid_search_even_if_another_store_supports_it(mock_vector_io_api):
    """Test that one unsupported store fails the whole search rather than the supported ones answering.

    Serving only the subset of the requested stores that can apply the weights would rank part of the
    answer one way, drop the rest, and tell the client nothing about either.
    """
    supported = VectorStoreSearchResponsePage(search_query=["receipts"], has_more=False, data=[])

    async def search(vector_store_id: str, request):
        if vector_store_id == "unsupported_store":
            raise _unsupported_hybrid_search_error("unsupported_store")
        return supported

    mock_vector_io_api.openai_search_vector_store.side_effect = search
    file_search_tool = OpenAIResponseInputToolFileSearch.model_validate(
        {
            "type": "file_search",
            "vector_store_ids": ["supported_store", "unsupported_store"],
            "ranking_options": {"hybrid_search": {"embedding_weight": 0.3, "text_weight": 0.7}},
        }
    )

    with pytest.raises(InvalidParameterError):
        await _tool_executor(mock_vector_io_api)._execute_file_search_via_vector_store(
            query="receipts",
            response_file_search_tool=file_search_tool,
        )


async def test_file_search_still_degrades_when_a_vector_store_fails(mock_vector_io_api, caplog):
    """Test that a store that is merely broken still costs only its own results, as before.

    Only client-parameter errors are surfaced; an unreachable backend keeps degrading to no
    results from that store so the remaining stores can still answer.
    """
    mock_vector_io_api.openai_search_vector_store.side_effect = ConnectionError("vector store is unreachable")

    with caplog.at_level("WARNING"):
        result = await _tool_executor(mock_vector_io_api)._execute_file_search_via_vector_store(
            query="What is the travel reimbursement limit?",
            response_file_search_tool=_hybrid_search_file_search_tool(),
        )

    assert result.error_message is None
    assert result.metadata["chunks"] == []
    assert result.metadata["document_ids"] == []
    assert "Failed to search vector store" in caplog.text


async def test_execute_tool_does_not_turn_unsupported_hybrid_search_into_a_tool_error(mock_vector_io_api):
    """Test that _execute_tool propagates the 400 instead of reporting it as a failed tool call.

    _execute_tool captures tool failures into error_exc, which the response reports as a failed
    file_search item inside an otherwise successful response. A rejected request parameter must
    not be reported that way.
    """
    mock_vector_io_api.openai_search_vector_store.side_effect = _unsupported_hybrid_search_error()
    tool_executor = _tool_executor(mock_vector_io_api)
    ctx = SimpleNamespace(response_tools=[_hybrid_search_file_search_tool()])

    with pytest.raises(InvalidParameterError):
        await tool_executor._execute_tool("file_search", {"query": "receipts"}, ctx)


async def test_responses_create_rejects_unsupported_hybrid_search(
    openai_responses_impl, mock_inference_api, mock_vector_io_api
):
    """Test that responses.create fails with a 400 when file_search hits an unsupported store.

    The whole request fails: the client gets Bad Request naming the parameter, not a 200 whose
    answer was written without the retrieval it asked for, and not a 500.
    """
    mock_vector_io_api.openai_search_vector_store.side_effect = _unsupported_hybrid_search_error()
    mock_inference_api.openai_chat_completion.side_effect = [fake_stream("file_search_tool_call_completion.yaml")]

    with pytest.raises(InvalidParameterError) as excinfo:
        await openai_responses_impl.create_openai_response(
            CreateResponseRequest(
                input="What is the travel reimbursement limit?",
                model="ollama/llama3.2:3b",
                tools=[_hybrid_search_file_search_tool()],
            )
        )

    http_exc = translate_exception(excinfo.value)
    assert http_exc.status_code == httpx.codes.BAD_REQUEST
    assert "ranking_options.hybrid_search" in http_exc.detail
    assert _unsupported_hybrid_search_message() in http_exc.detail


async def test_responses_create_streaming_reports_unsupported_hybrid_search_as_error_event(
    openai_responses_impl, mock_inference_api, mock_vector_io_api
):
    """Test what a stream=True client sees: a terminal error event carrying the 400.

    The HTTP status is already 200 by the time the stream starts, so the SSE layer reports the
    rejection as the final event. Its code is the 400 and its message is the parameter error
    itself, not the "Internal server error" text reserved for 5xx.
    """
    mock_vector_io_api.openai_search_vector_store.side_effect = _unsupported_hybrid_search_error()
    mock_inference_api.openai_chat_completion.side_effect = [fake_stream("file_search_tool_call_completion.yaml")]

    stream = await openai_responses_impl.create_openai_response(
        CreateResponseRequest(
            input="What is the travel reimbursement limit?",
            model="ollama/llama3.2:3b",
            tools=[_hybrid_search_file_search_tool()],
            stream=True,
        )
    )

    events = [json.loads(chunk.removeprefix("data: ")) async for chunk in sse_generator(stream)]

    assert events[-1]["type"] == "error"
    assert events[-1]["code"] == "400"
    assert "ranking_options.hybrid_search" in events[-1]["message"]
    assert _unsupported_hybrid_search_message() in events[-1]["message"]
    # No terminal lifecycle event claims the response completed or merely failed on the server.
    assert not [event for event in events if event["type"] in {"response.completed", "response.failed"}]


async def test_background_responses_create_records_unsupported_hybrid_search_as_failed(
    openai_responses_impl, mock_inference_api, mock_vector_io_api, mock_responses_store
):
    """Test what a background=True client polling the response sees: a failed response, not a stuck one.

    The request itself has already returned "queued", so the rejection can only reach the caller
    through the stored response, which the background worker marks failed with the parameter error.
    """
    mock_vector_io_api.openai_search_vector_store.side_effect = _unsupported_hybrid_search_error()
    mock_inference_api.openai_chat_completion.side_effect = [fake_stream("file_search_tool_call_completion.yaml")]
    mock_responses_store.get_response_object.return_value = _OpenAIResponseObjectWithInputAndMessages(
        id="resp_background",
        created_at=1234567890,
        model="ollama/llama3.2:3b",
        status="in_progress",
        output=[],
        input=[],
        store=True,
    )

    queued = await openai_responses_impl.create_openai_response(
        CreateResponseRequest(
            input="What is the travel reimbursement limit?",
            model="ollama/llama3.2:3b",
            tools=[_hybrid_search_file_search_tool()],
            background=True,
        )
    )
    assert queued.status == "queued"

    try:
        await asyncio.wait_for(openai_responses_impl._background_queue.join(), timeout=10)
    finally:
        await openai_responses_impl.shutdown()

    stored = [call.args[0] for call in mock_responses_store.update_response_object.call_args_list]
    failed = [response for response in stored if response.status == "failed"]
    assert failed, "the background worker should have stored a failed response"
    assert failed[-1].error.code == "processing_error"
    assert "ranking_options.hybrid_search" in failed[-1].error.message
    assert _unsupported_hybrid_search_message() in failed[-1].error.message
