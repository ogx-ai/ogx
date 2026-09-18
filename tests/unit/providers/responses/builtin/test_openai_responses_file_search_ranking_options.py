# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import httpx
import pytest
from pydantic import ValidationError

from ogx.core.datatypes import VectorStoresConfig
from ogx.providers.inline.responses.builtin.responses.tool_executor import ToolExecutor
from ogx_api.common.errors import InvalidParameterError
from ogx_api.openai_responses import OpenAIResponseInputToolFileSearch
from ogx_api.vector_io import SearchRankingOptions, VectorStoreSearchResponsePage


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


async def test_file_search_swallows_unsupported_hybrid_search_error(mock_vector_io_api, caplog):
    """Test today's Responses behaviour when the vector store rejects hybrid_search.

    openai_search_vector_store raises InvalidParameterError (HTTP 400) on a provider that cannot
    apply the weights, but ToolExecutor._execute_file_search_via_vector_store catches every
    exception per vector store and returns no results, so the caller sees an empty file_search
    result instead of the 400. This test records that behaviour rather than endorsing it;
    changing the tool executor is out of scope for this change.
    """
    query = "What is the travel reimbursement limit?"
    error = InvalidParameterError(
        "ranking_options.hybrid_search",
        {"embedding_weight": 0.3, "text_weight": 0.7},
        "The provider of vector store 'test_vector_store' does not support weighted hybrid search.",
    )
    assert error.status_code == httpx.codes.BAD_REQUEST
    mock_vector_io_api.openai_search_vector_store.side_effect = error

    tool_executor = ToolExecutor(
        tool_groups_api=None,  # type: ignore
        tool_runtime_api=None,  # type: ignore
        vector_io_api=mock_vector_io_api,
        vector_stores_config=VectorStoresConfig(),
        mcp_session_manager=None,
    )
    file_search_tool = OpenAIResponseInputToolFileSearch.model_validate(
        {
            "type": "file_search",
            "vector_store_ids": ["test_vector_store"],
            "ranking_options": {"hybrid_search": {"embedding_weight": 0.3, "text_weight": 0.7}},
        }
    )

    with caplog.at_level("WARNING"):
        result = await tool_executor._execute_file_search_via_vector_store(
            query=query,
            response_file_search_tool=file_search_tool,
        )

    # The 400 never reaches the client: it is logged and turned into zero search results.
    assert result.error_message is None
    assert result.metadata["chunks"] == []
    assert result.metadata["document_ids"] == []
    assert "Failed to search vector store" in caplog.text
