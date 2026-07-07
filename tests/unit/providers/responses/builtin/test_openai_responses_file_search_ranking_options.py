# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from ogx.core.datatypes import VectorStoresConfig
from ogx.providers.inline.responses.builtin.responses.tool_executor import ToolExecutor
from ogx_api.openai_responses import OpenAIResponseInputToolFileSearch
from ogx_api.vector_io import SearchRankingOptions, VectorStoreSearchResponsePage


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
