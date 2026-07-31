# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import json
import time

import pytest

from .helpers import assert_text_contains, new_vector_store, skip_if_provider_is_vertexai, upload_file


@pytest.mark.parametrize(
    "text_format",
    # Not testing json_object because most providers don't actually support it.
    [
        {"type": "text"},
        {
            "type": "json_schema",
            "name": "capitals",
            "description": "A schema for the capital of each country",
            "schema": {"type": "object", "properties": {"capital": {"type": "string"}}},
            "strict": True,
        },
    ],
)
def test_response_text_format(responses_client, text_model_id, text_format):
    if text_format.get("type") == "json_schema" and "watsonx" in text_model_id:
        pytest.skip("WatsonX does not reliably follow json_schema output format.")
    stream = False
    response = responses_client.responses.create(
        model=text_model_id,
        input="What is the capital of France?",
        stream=stream,
        text={"format": text_format},
    )
    # by_alias=True is needed because otherwise Pydantic renames our "schema" field
    assert response.text.format.model_dump(exclude_none=True, by_alias=True) == text_format
    assert_text_contains(response.output_text, "paris")
    if text_format["type"] == "json_schema":
        assert_text_contains(json.loads(response.output_text)["capital"], "paris")


@pytest.fixture
def vector_store_with_filtered_files(
    request, responses_client, client_with_models, embedding_model_id, embedding_dimension, tmp_path_factory
):
    # """Create a vector store with multiple files that have different attributes for filtering tests."""
    if "text_model_id" in request.fixturenames:
        text_model_id = request.getfixturevalue("text_model_id")
        if text_model_id and text_model_id.startswith("watsonx/"):
            pytest.skip("WatsonX file search filters are not reliably supported")
        if text_model_id:
            skip_if_provider_is_vertexai(
                client_with_models,
                text_model_id,
                "file search filter replay fails due to non-deterministic vector store IDs",
            )
    vector_store = new_vector_store(
        responses_client, "test_vector_store_with_filters", embedding_model_id, embedding_dimension
    )
    tmp_path = tmp_path_factory.mktemp("filter_test_files")

    # Create multiple files with different attributes
    files_data = [
        {
            "name": "us_marketing_q1.txt",
            "content": "US promotional campaigns for Q1 2023. Revenue increased by 15% in the US region.",
            "attributes": {
                "region": "us",
                "category": "marketing",
                "date": 1672531200,  # Jan 1, 2023
            },
        },
        {
            "name": "us_engineering_q2.txt",
            "content": "US technical updates for Q2 2023. New features deployed in the US region.",
            "attributes": {
                "region": "us",
                "category": "engineering",
                "date": 1680307200,  # Apr 1, 2023
            },
        },
        {
            "name": "eu_marketing_q1.txt",
            "content": "European advertising campaign results for Q1 2023. Strong growth in EU markets.",
            "attributes": {
                "region": "eu",
                "category": "marketing",
                "date": 1672531200,  # Jan 1, 2023
            },
        },
        {
            "name": "asia_sales_q3.txt",
            "content": "Asia Pacific revenue figures for Q3 2023. Record breaking quarter in Asia.",
            "attributes": {
                "region": "asia",
                "category": "sales",
                "date": 1688169600,  # Jul 1, 2023
            },
        },
    ]

    file_ids = []
    for file_data in files_data:
        # Create file
        file_path = tmp_path / file_data["name"]
        file_path.write_text(file_data["content"])

        # Upload file
        file_response = upload_file(responses_client, file_data["name"], str(file_path))
        file_ids.append(file_response.id)

        # Attach file to vector store with attributes
        file_attach_response = responses_client.vector_stores.files.create(
            vector_store_id=vector_store.id,
            file_id=file_response.id,
            attributes=file_data["attributes"],
        )

        # Wait for attachment
        while file_attach_response.status == "in_progress":
            time.sleep(0.1)
            file_attach_response = responses_client.vector_stores.files.retrieve(
                vector_store_id=vector_store.id,
                file_id=file_response.id,
            )
        assert file_attach_response.status == "completed"

    yield vector_store

    # Cleanup: delete vector store and files
    try:
        responses_client.vector_stores.delete(vector_store_id=vector_store.id)
        for file_id in file_ids:
            try:
                responses_client.files.delete(file_id=file_id)
            except Exception:
                pass  # File might already be deleted
    except Exception:
        pass  # Best effort cleanup


def test_response_file_search_filter_by_region(
    responses_client, client_with_models, text_model_id, vector_store_with_filtered_files
):
    """Test file search with region equality filter."""
    if text_model_id.startswith("watsonx/"):
        pytest.skip("WatsonX file search filters are not reliably supported")
    skip_if_provider_is_vertexai(
        client_with_models, text_model_id, "file search filter replay fails due to non-deterministic vector store IDs"
    )
    tools = [
        {
            "type": "file_search",
            "vector_store_ids": [vector_store_with_filtered_files.id],
            "filters": {"type": "eq", "key": "region", "value": "us"},
        }
    ]

    response = responses_client.responses.create(
        model=text_model_id,
        input="What are the updates from the US region?",
        tools=tools,
        stream=False,
        include=["file_search_call.results"],
    )

    # Verify file search was called with US filter
    assert len(response.output) > 1
    assert response.output[0].type == "file_search_call"
    assert response.output[0].status == "completed"
    assert response.output[0].results
    # Should only return US files (not EU or Asia files)
    for result in response.output[0].results:
        assert "us" in result.text.lower() or "US" in result.text
        # Ensure non-US regions are NOT returned
        assert "european" not in result.text.lower()
        assert "asia" not in result.text.lower()


def test_response_file_search_filter_by_category(
    responses_client, client_with_models, text_model_id, vector_store_with_filtered_files
):
    """Test file search with category equality filter."""
    if text_model_id.startswith("watsonx/"):
        pytest.skip("WatsonX does not reliably support tool calling")
    skip_if_provider_is_vertexai(
        client_with_models, text_model_id, "file search filter behavior differs from expected results"
    )
    tools = [
        {
            "type": "file_search",
            "vector_store_ids": [vector_store_with_filtered_files.id],
            "filters": {"type": "eq", "key": "category", "value": "marketing"},
        }
    ]

    response = responses_client.responses.create(
        model=text_model_id,
        input="Show me all marketing reports",
        tools=tools,
        stream=False,
        include=["file_search_call.results"],
    )

    assert response.output[0].type == "file_search_call"
    assert response.output[0].status == "completed"
    assert response.output[0].results
    # Should only return marketing files (not engineering or sales)
    for result in response.output[0].results:
        # Marketing files should have promotional/advertising content
        assert "promotional" in result.text.lower() or "advertising" in result.text.lower()
        # Ensure non-marketing categories are NOT returned
        assert "technical" not in result.text.lower()
        assert "revenue figures" not in result.text.lower()


def test_response_file_search_filter_by_date_range(
    responses_client, client_with_models, text_model_id, vector_store_with_filtered_files
):
    """Test file search with date range filter using compound AND."""
    if text_model_id.startswith("watsonx/"):
        pytest.skip("WatsonX file search filters are not reliably supported")
    skip_if_provider_is_vertexai(
        client_with_models, text_model_id, "file search filter replay fails due to non-deterministic vector store IDs"
    )
    tools = [
        {
            "type": "file_search",
            "vector_store_ids": [vector_store_with_filtered_files.id],
            "filters": {
                "type": "and",
                "filters": [
                    {
                        "type": "gte",
                        "key": "date",
                        "value": 1672531200,  # Jan 1, 2023
                    },
                    {
                        "type": "lt",
                        "key": "date",
                        "value": 1680307200,  # Apr 1, 2023
                    },
                ],
            },
        }
    ]

    response = responses_client.responses.create(
        model=text_model_id,
        input="What happened in Q1 2023?",
        tools=tools,
        stream=False,
        include=["file_search_call.results"],
    )

    assert response.output[0].type == "file_search_call"
    assert response.output[0].status == "completed"
    assert response.output[0].results
    # Should only return Q1 files (not Q2 or Q3)
    for result in response.output[0].results:
        assert "q1" in result.text.lower()
        # Ensure non-Q1 quarters are NOT returned
        assert "q2" not in result.text.lower()
        assert "q3" not in result.text.lower()


def test_response_file_search_filter_compound_and(
    responses_client, client_with_models, text_model_id, vector_store_with_filtered_files
):
    """Test file search with compound AND filter (region AND category)."""
    if text_model_id.startswith("watsonx/"):
        pytest.skip("WatsonX file search filters are not reliably supported")
    skip_if_provider_is_vertexai(
        client_with_models, text_model_id, "file search filter replay fails due to non-deterministic vector store IDs"
    )
    tools = [
        {
            "type": "file_search",
            "vector_store_ids": [vector_store_with_filtered_files.id],
            "filters": {
                "type": "and",
                "filters": [
                    {"type": "eq", "key": "region", "value": "us"},
                    {"type": "eq", "key": "category", "value": "engineering"},
                ],
            },
        }
    ]

    response = responses_client.responses.create(
        model=text_model_id,
        input="What are the engineering updates from the US?",
        tools=tools,
        stream=False,
        include=["file_search_call.results"],
    )

    assert response.output[0].type == "file_search_call"
    assert response.output[0].status == "completed"
    assert response.output[0].results
    # Should only return US engineering files
    assert len(response.output[0].results) >= 1
    for result in response.output[0].results:
        assert "us" in result.text.lower() and "technical" in result.text.lower()
        # Ensure it's not from other regions or categories
        assert "european" not in result.text.lower() and "asia" not in result.text.lower()
        assert "promotional" not in result.text.lower() and "revenue" not in result.text.lower()


def test_response_file_search_filter_compound_or(responses_client, text_model_id, vector_store_with_filtered_files):
    """Test file search with compound OR filter (marketing OR sales)."""
    if text_model_id.startswith("watsonx/"):
        pytest.skip("WatsonX does not reliably support tool calling")
    tools = [
        {
            "type": "file_search",
            "vector_store_ids": [vector_store_with_filtered_files.id],
            "filters": {
                "type": "or",
                "filters": [
                    {"type": "eq", "key": "category", "value": "marketing"},
                    {"type": "eq", "key": "category", "value": "sales"},
                ],
            },
        }
    ]

    response = responses_client.responses.create(
        model=text_model_id,
        input="Show me marketing and sales documents",
        tools=tools,
        stream=False,
        include=["file_search_call.results"],
    )

    assert response.output[0].type == "file_search_call"
    assert response.output[0].status == "completed"
    assert response.output[0].results
    # Should return marketing and sales files, but NOT engineering
    categories_found = set()
    for result in response.output[0].results:
        text_lower = result.text.lower()
        if "promotional" in text_lower or "advertising" in text_lower:
            categories_found.add("marketing")
        if "revenue figures" in text_lower:
            categories_found.add("sales")
        # Ensure engineering files are NOT returned
        assert "technical" not in text_lower, f"Engineering file should not be returned, but got: {result.text}"

    # Verify we got at least one of the expected categories
    assert len(categories_found) > 0, "Should have found at least one marketing or sales file"
    assert categories_found.issubset({"marketing", "sales"}), f"Found unexpected categories: {categories_found}"


def test_response_file_search_streaming_events(responses_client, text_model_id, vector_store_with_filtered_files):
    """Test that file search emits proper streaming events (in_progress, searching, completed)."""
    if text_model_id.startswith("watsonx/"):
        pytest.skip("WatsonX does not reliably support tool calling")
    tools = [
        {
            "type": "file_search",
            "vector_store_ids": [vector_store_with_filtered_files.id],
        }
    ]

    stream = responses_client.responses.create(
        model=text_model_id,
        input="What are the marketing updates?",
        tools=tools,
        stream=True,
    )

    chunks = []
    for chunk in stream:
        chunks.append(chunk)

    event_types = [chunk.type for chunk in chunks]

    # Verify file search streaming events are present
    file_search_in_progress = [chunk for chunk in chunks if chunk.type == "response.file_search_call.in_progress"]
    file_search_searching = [chunk for chunk in chunks if chunk.type == "response.file_search_call.searching"]
    file_search_completed = [chunk for chunk in chunks if chunk.type == "response.file_search_call.completed"]

    assert len(file_search_in_progress) > 0, (
        f"Expected response.file_search_call.in_progress events, got chunk types: {event_types}"
    )
    assert len(file_search_searching) > 0, (
        f"Expected response.file_search_call.searching events, got chunk types: {event_types}"
    )
    assert len(file_search_completed) > 0, (
        f"Expected response.file_search_call.completed events, got chunk types: {event_types}"
    )

    # Verify final response has file search call
    final_chunk = chunks[-1]
    if hasattr(final_chunk, "response"):
        file_search_calls = [output for output in final_chunk.response.output if output.type == "file_search_call"]
        assert len(file_search_calls) > 0, "Expected at least one file_search_call in final response"
        assert file_search_calls[0].status == "completed"
