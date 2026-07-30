# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""
Integration tests for OpenAI-compatible filter functionality in vector stores.

These tests verify that filter operations (comparison and compound) work correctly
across different vector store providers (sqlite-vec, faiss).

Note: Some providers (chroma, qdrant, weaviate, elasticsearch) do not yet support native
filtering and will be skipped in these tests.
"""

import time

import pytest

from ogx_api import ChunkMetadata, EmbeddedChunk
from ogx_api.filters import ComparisonFilter, CompoundFilter

from ..conftest import vector_provider_wrapper

# Providers that support native filtering
# Other providers (chroma, qdrant, weaviate, elasticsearch) raise NotImplementedError
PROVIDERS_WITH_NATIVE_FILTERING = {"faiss", "sqlite-vec", "milvus", "pgvector"}


def skip_if_provider_doesnt_support_filters(vector_io_provider_id: str):
    """Skip test if the provider doesn't support native filtering."""
    # Extract provider name from provider_id which may be like "inline::faiss" or just "faiss"
    provider_name = vector_io_provider_id.split("::")[-1] if "::" in vector_io_provider_id else vector_io_provider_id
    if provider_name not in PROVIDERS_WITH_NATIVE_FILTERING:
        pytest.skip(
            f"Provider '{provider_name}' does not yet support native filtering. "
            f"Supported providers: {PROVIDERS_WITH_NATIVE_FILTERING}"
        )


@pytest.fixture(scope="session")
def filter_test_chunks(embedding_dimension):
    """Create test chunks with varied metadata for filter testing."""
    import numpy as np

    from ogx.providers.utils.vector_io.vector_utils import generate_chunk_id

    # Create chunks with diverse metadata for comprehensive filter testing
    chunks_data = [
        {
            "content": "Python is a high-level programming language known for its readability.",
            "document_id": "doc1",
            "topic": "programming",
            "category": "languages",
            "priority": 1,
            "is_featured": True,
            "year": 2020,
        },
        {
            "content": "Machine learning enables systems to learn from data automatically.",
            "document_id": "doc2",
            "topic": "ai",
            "category": "technology",
            "priority": 2,
            "is_featured": True,
            "year": 2021,
        },
        {
            "content": "Data structures provide efficient ways to organize and access data.",
            "document_id": "doc3",
            "topic": "computer_science",
            "category": "fundamentals",
            "priority": 3,
            "is_featured": False,
            "year": 2019,
        },
        {
            "content": "Neural networks are inspired by the biological neural networks in brains.",
            "document_id": "doc4",
            "topic": "ai",
            "category": "technology",
            "priority": 1,
            "is_featured": True,
            "year": 2022,
        },
        {
            "content": "JavaScript is the language of the web, running in browsers and servers.",
            "document_id": "doc5",
            "topic": "programming",
            "category": "languages",
            "priority": 2,
            "is_featured": False,
            "year": 2018,
        },
    ]

    np.random.seed(42)

    embedded_chunks = []
    for data in chunks_data:
        content = data["content"]
        doc_id = data["document_id"]
        chunk_id = generate_chunk_id(doc_id, content)
        embedding = np.random.random(int(embedding_dimension)).tolist()

        metadata = {
            "document_id": doc_id,
            "topic": data["topic"],
            "category": data["category"],
            "priority": data["priority"],
            "is_featured": data["is_featured"],
            "year": data["year"],
        }

        embedded_chunk = EmbeddedChunk(
            content=content,
            chunk_id=chunk_id,
            metadata=metadata,
            chunk_metadata=ChunkMetadata(
                document_id=doc_id,
                chunk_id=chunk_id,
                created_timestamp=int(time.time()),
                updated_timestamp=int(time.time()),
                content_token_count=len(content.split()),
            ),
            embedding=embedding,
            embedding_model="test-embedding-model",
            embedding_dimension=int(embedding_dimension),
        )
        embedded_chunks.append(embedded_chunk)

    return embedded_chunks


@pytest.fixture(scope="function")
def client_with_empty_registry(client_with_models):
    """Fixture that provides a client with cleared vector stores."""

    def clear_registry():
        vector_stores = client_with_models.vector_stores.list()
        for vector_store in vector_stores.data:
            client_with_models.vector_stores.delete(vector_store_id=vector_store.id)

    clear_registry()
    yield client_with_models
    clear_registry()


# =============================================================================
# Comparison Filter Tests
# =============================================================================


@vector_provider_wrapper
def test_filter_eq_string(
    client_with_empty_registry,
    filter_test_chunks,
    embedding_model_id,
    embedding_dimension,
    vector_io_provider_id,
):
    """Test equality filter on string field."""
    skip_if_provider_doesnt_support_filters(vector_io_provider_id)
    client = client_with_empty_registry

    vector_store = client.vector_stores.create(
        name="filter_eq_test",
        extra_body={
            "provider_id": vector_io_provider_id,
            "embedding_model": embedding_model_id,
        },
    )

    client.vector_io.insert(
        vector_store_id=vector_store.id,
        chunks=filter_test_chunks,
    )

    # Filter for topic == "ai"
    filter_obj = ComparisonFilter(type="eq", key="topic", value="ai")

    response = client.vector_io.query(
        vector_store_id=vector_store.id,
        query="technology",
        params={"filters": filter_obj.model_dump()},
    )

    assert response is not None
    assert len(response.chunks) > 0

    # All results should have topic "ai"
    for chunk in response.chunks:
        assert chunk.metadata["topic"] == "ai"


@vector_provider_wrapper
def test_filter_ne_string(
    client_with_empty_registry,
    filter_test_chunks,
    embedding_model_id,
    embedding_dimension,
    vector_io_provider_id,
):
    """Test not-equal filter on string field."""
    skip_if_provider_doesnt_support_filters(vector_io_provider_id)
    client = client_with_empty_registry

    vector_store = client.vector_stores.create(
        name="filter_ne_test",
        extra_body={
            "provider_id": vector_io_provider_id,
            "embedding_model": embedding_model_id,
        },
    )

    client.vector_io.insert(
        vector_store_id=vector_store.id,
        chunks=filter_test_chunks,
    )

    # Filter for topic != "ai"
    filter_obj = ComparisonFilter(type="ne", key="topic", value="ai")

    response = client.vector_io.query(
        vector_store_id=vector_store.id,
        query="programming",
        params={"filters": filter_obj.model_dump()},
    )

    assert response is not None
    assert len(response.chunks) > 0

    # No results should have topic "ai"
    for chunk in response.chunks:
        assert chunk.metadata["topic"] != "ai"


@vector_provider_wrapper
def test_filter_gt_numeric(
    client_with_empty_registry,
    filter_test_chunks,
    embedding_model_id,
    embedding_dimension,
    vector_io_provider_id,
):
    """Test greater-than filter on numeric field."""
    skip_if_provider_doesnt_support_filters(vector_io_provider_id)
    client = client_with_empty_registry

    vector_store = client.vector_stores.create(
        name="filter_gt_test",
        extra_body={
            "provider_id": vector_io_provider_id,
            "embedding_model": embedding_model_id,
        },
    )

    client.vector_io.insert(
        vector_store_id=vector_store.id,
        chunks=filter_test_chunks,
    )

    # Filter for year > 2020
    filter_obj = ComparisonFilter(type="gt", key="year", value=2020)

    response = client.vector_io.query(
        vector_store_id=vector_store.id,
        query="technology",
        params={"filters": filter_obj.model_dump()},
    )

    assert response is not None
    assert len(response.chunks) > 0

    # All results should have year > 2020
    for chunk in response.chunks:
        assert chunk.metadata["year"] > 2020


@vector_provider_wrapper
def test_filter_gte_numeric(
    client_with_empty_registry,
    filter_test_chunks,
    embedding_model_id,
    embedding_dimension,
    vector_io_provider_id,
):
    """Test greater-than-or-equal filter on numeric field."""
    skip_if_provider_doesnt_support_filters(vector_io_provider_id)
    client = client_with_empty_registry

    vector_store = client.vector_stores.create(
        name="filter_gte_test",
        extra_body={
            "provider_id": vector_io_provider_id,
            "embedding_model": embedding_model_id,
        },
    )

    client.vector_io.insert(
        vector_store_id=vector_store.id,
        chunks=filter_test_chunks,
    )

    # Filter for priority >= 2
    filter_obj = ComparisonFilter(type="gte", key="priority", value=2)

    response = client.vector_io.query(
        vector_store_id=vector_store.id,
        query="technology",
        params={"filters": filter_obj.model_dump()},
    )

    assert response is not None
    assert len(response.chunks) > 0

    # All results should have priority >= 2
    for chunk in response.chunks:
        assert chunk.metadata["priority"] >= 2


@vector_provider_wrapper
def test_filter_lt_numeric(
    client_with_empty_registry,
    filter_test_chunks,
    embedding_model_id,
    embedding_dimension,
    vector_io_provider_id,
):
    """Test less-than filter on numeric field."""
    skip_if_provider_doesnt_support_filters(vector_io_provider_id)
    client = client_with_empty_registry

    vector_store = client.vector_stores.create(
        name="filter_lt_test",
        extra_body={
            "provider_id": vector_io_provider_id,
            "embedding_model": embedding_model_id,
        },
    )

    client.vector_io.insert(
        vector_store_id=vector_store.id,
        chunks=filter_test_chunks,
    )

    # Filter for year < 2020
    filter_obj = ComparisonFilter(type="lt", key="year", value=2020)

    response = client.vector_io.query(
        vector_store_id=vector_store.id,
        query="technology",
        params={"filters": filter_obj.model_dump()},
    )

    assert response is not None
    assert len(response.chunks) > 0

    # All results should have year < 2020
    for chunk in response.chunks:
        assert chunk.metadata["year"] < 2020


@vector_provider_wrapper
def test_filter_lte_numeric(
    client_with_empty_registry,
    filter_test_chunks,
    embedding_model_id,
    embedding_dimension,
    vector_io_provider_id,
):
    """Test less-than-or-equal filter on numeric field."""
    skip_if_provider_doesnt_support_filters(vector_io_provider_id)
    client = client_with_empty_registry

    vector_store = client.vector_stores.create(
        name="filter_lte_test",
        extra_body={
            "provider_id": vector_io_provider_id,
            "embedding_model": embedding_model_id,
        },
    )

    client.vector_io.insert(
        vector_store_id=vector_store.id,
        chunks=filter_test_chunks,
    )

    # Filter for priority <= 2
    filter_obj = ComparisonFilter(type="lte", key="priority", value=2)

    response = client.vector_io.query(
        vector_store_id=vector_store.id,
        query="technology",
        params={"filters": filter_obj.model_dump()},
    )

    assert response is not None
    assert len(response.chunks) > 0

    # All results should have priority <= 2
    for chunk in response.chunks:
        assert chunk.metadata["priority"] <= 2


@vector_provider_wrapper
def test_filter_in_list(
    client_with_empty_registry,
    filter_test_chunks,
    embedding_model_id,
    embedding_dimension,
    vector_io_provider_id,
):
    """Test 'in' filter with list of values."""
    skip_if_provider_doesnt_support_filters(vector_io_provider_id)
    client = client_with_empty_registry

    vector_store = client.vector_stores.create(
        name="filter_in_test",
        extra_body={
            "provider_id": vector_io_provider_id,
            "embedding_model": embedding_model_id,
        },
    )

    client.vector_io.insert(
        vector_store_id=vector_store.id,
        chunks=filter_test_chunks,
    )

    # Filter for topic in ["ai", "programming"]
    filter_obj = ComparisonFilter(type="in", key="topic", value=["ai", "programming"])

    response = client.vector_io.query(
        vector_store_id=vector_store.id,
        query="technology",
        params={"filters": filter_obj.model_dump()},
    )

    assert response is not None
    assert len(response.chunks) > 0

    # All results should have topic "ai" or "programming"
    for chunk in response.chunks:
        assert chunk.metadata["topic"] in ["ai", "programming"]


@vector_provider_wrapper
def test_filter_nin_list(
    client_with_empty_registry,
    filter_test_chunks,
    embedding_model_id,
    embedding_dimension,
    vector_io_provider_id,
):
    """Test 'not in' filter with list of values."""
    skip_if_provider_doesnt_support_filters(vector_io_provider_id)
    client = client_with_empty_registry

    vector_store = client.vector_stores.create(
        name="filter_nin_test",
        extra_body={
            "provider_id": vector_io_provider_id,
            "embedding_model": embedding_model_id,
        },
    )

    client.vector_io.insert(
        vector_store_id=vector_store.id,
        chunks=filter_test_chunks,
    )

    # Filter for topic not in ["ai", "programming"]
    filter_obj = ComparisonFilter(type="nin", key="topic", value=["ai", "programming"])

    response = client.vector_io.query(
        vector_store_id=vector_store.id,
        query="fundamentals",
        params={"filters": filter_obj.model_dump()},
    )

    assert response is not None
    assert len(response.chunks) > 0

    # No results should have topic "ai" or "programming"
    for chunk in response.chunks:
        assert chunk.metadata["topic"] not in ["ai", "programming"]


@vector_provider_wrapper
def test_filter_eq_boolean(
    client_with_empty_registry,
    filter_test_chunks,
    embedding_model_id,
    embedding_dimension,
    vector_io_provider_id,
):
    """Test equality filter on boolean field."""
    skip_if_provider_doesnt_support_filters(vector_io_provider_id)
    client = client_with_empty_registry

    vector_store = client.vector_stores.create(
        name="filter_bool_test",
        extra_body={
            "provider_id": vector_io_provider_id,
            "embedding_model": embedding_model_id,
        },
    )

    client.vector_io.insert(
        vector_store_id=vector_store.id,
        chunks=filter_test_chunks,
    )

    # Filter for is_featured == True
    filter_obj = ComparisonFilter(type="eq", key="is_featured", value=True)

    response = client.vector_io.query(
        vector_store_id=vector_store.id,
        query="technology",
        params={"filters": filter_obj.model_dump()},
    )

    assert response is not None
    assert len(response.chunks) > 0

    # All results should have is_featured == True
    for chunk in response.chunks:
        assert chunk.metadata["is_featured"] is True


# =============================================================================
# Compound Filter Tests
# =============================================================================


@vector_provider_wrapper
def test_filter_and_compound(
    client_with_empty_registry,
    filter_test_chunks,
    embedding_model_id,
    embedding_dimension,
    vector_io_provider_id,
):
    """Test AND compound filter combining multiple conditions."""
    skip_if_provider_doesnt_support_filters(vector_io_provider_id)
    client = client_with_empty_registry

    vector_store = client.vector_stores.create(
        name="filter_and_test",
        extra_body={
            "provider_id": vector_io_provider_id,
            "embedding_model": embedding_model_id,
        },
    )

    client.vector_io.insert(
        vector_store_id=vector_store.id,
        chunks=filter_test_chunks,
    )

    # Filter for topic == "ai" AND is_featured == True
    filter_obj = CompoundFilter(
        type="and",
        filters=[
            ComparisonFilter(type="eq", key="topic", value="ai"),
            ComparisonFilter(type="eq", key="is_featured", value=True),
        ],
    )

    response = client.vector_io.query(
        vector_store_id=vector_store.id,
        query="technology",
        params={"filters": filter_obj.model_dump()},
    )

    assert response is not None
    assert len(response.chunks) > 0

    # All results should match both conditions
    for chunk in response.chunks:
        assert chunk.metadata["topic"] == "ai"
        assert chunk.metadata["is_featured"] is True


@vector_provider_wrapper
def test_filter_or_compound(
    client_with_empty_registry,
    filter_test_chunks,
    embedding_model_id,
    embedding_dimension,
    vector_io_provider_id,
):
    """Test OR compound filter combining multiple conditions."""
    skip_if_provider_doesnt_support_filters(vector_io_provider_id)
    client = client_with_empty_registry

    vector_store = client.vector_stores.create(
        name="filter_or_test",
        extra_body={
            "provider_id": vector_io_provider_id,
            "embedding_model": embedding_model_id,
        },
    )

    client.vector_io.insert(
        vector_store_id=vector_store.id,
        chunks=filter_test_chunks,
    )

    # Filter for topic == "ai" OR category == "languages"
    filter_obj = CompoundFilter(
        type="or",
        filters=[
            ComparisonFilter(type="eq", key="topic", value="ai"),
            ComparisonFilter(type="eq", key="category", value="languages"),
        ],
    )

    response = client.vector_io.query(
        vector_store_id=vector_store.id,
        query="technology",
        params={"filters": filter_obj.model_dump()},
    )

    assert response is not None
    assert len(response.chunks) > 0

    # All results should match at least one condition
    for chunk in response.chunks:
        matches_topic = chunk.metadata["topic"] == "ai"
        matches_category = chunk.metadata["category"] == "languages"
        assert matches_topic or matches_category


@vector_provider_wrapper
def test_filter_nested_compound(
    client_with_empty_registry,
    filter_test_chunks,
    embedding_model_id,
    embedding_dimension,
    vector_io_provider_id,
):
    """Test nested compound filters with complex logic."""
    skip_if_provider_doesnt_support_filters(vector_io_provider_id)
    client = client_with_empty_registry

    vector_store = client.vector_stores.create(
        name="filter_nested_test",
        extra_body={
            "provider_id": vector_io_provider_id,
            "embedding_model": embedding_model_id,
        },
    )

    client.vector_io.insert(
        vector_store_id=vector_store.id,
        chunks=filter_test_chunks,
    )

    # Complex filter: (topic == "ai" AND year >= 2021) OR (category == "languages")
    filter_obj = CompoundFilter(
        type="or",
        filters=[
            CompoundFilter(
                type="and",
                filters=[
                    ComparisonFilter(type="eq", key="topic", value="ai"),
                    ComparisonFilter(type="gte", key="year", value=2021),
                ],
            ),
            ComparisonFilter(type="eq", key="category", value="languages"),
        ],
    )

    response = client.vector_io.query(
        vector_store_id=vector_store.id,
        query="technology",
        params={"filters": filter_obj.model_dump()},
    )

    assert response is not None
    assert len(response.chunks) > 0

    # All results should match the complex condition
    for chunk in response.chunks:
        ai_and_recent = chunk.metadata["topic"] == "ai" and chunk.metadata["year"] >= 2021
        is_language = chunk.metadata["category"] == "languages"
        assert ai_and_recent or is_language


# =============================================================================
# Edge Case Tests
# =============================================================================


@vector_provider_wrapper
def test_filter_no_matches(
    client_with_empty_registry,
    filter_test_chunks,
    embedding_model_id,
    embedding_dimension,
    vector_io_provider_id,
):
    """Test filter that matches no documents."""
    skip_if_provider_doesnt_support_filters(vector_io_provider_id)
    client = client_with_empty_registry

    vector_store = client.vector_stores.create(
        name="filter_no_match_test",
        extra_body={
            "provider_id": vector_io_provider_id,
            "embedding_model": embedding_model_id,
        },
    )

    client.vector_io.insert(
        vector_store_id=vector_store.id,
        chunks=filter_test_chunks,
    )

    # Filter for topic that doesn't exist
    filter_obj = ComparisonFilter(type="eq", key="topic", value="nonexistent_topic")

    response = client.vector_io.query(
        vector_store_id=vector_store.id,
        query="technology",
        params={"filters": filter_obj.model_dump()},
    )

    assert response is not None
    assert len(response.chunks) == 0


@vector_provider_wrapper
def test_filter_null_returns_all(
    client_with_empty_registry,
    filter_test_chunks,
    embedding_model_id,
    embedding_dimension,
    vector_io_provider_id,
):
    """Test that None filter returns all matching results."""
    skip_if_provider_doesnt_support_filters(vector_io_provider_id)
    client = client_with_empty_registry

    vector_store = client.vector_stores.create(
        name="filter_null_test",
        extra_body={
            "provider_id": vector_io_provider_id,
            "embedding_model": embedding_model_id,
        },
    )

    client.vector_io.insert(
        vector_store_id=vector_store.id,
        chunks=filter_test_chunks,
    )

    # Query without filter
    response = client.vector_io.query(
        vector_store_id=vector_store.id,
        query="technology programming",
    )

    assert response is not None
    # Should return results without filtering
    assert len(response.chunks) > 0


@vector_provider_wrapper
def test_filter_multiple_and_conditions(
    client_with_empty_registry,
    filter_test_chunks,
    embedding_model_id,
    embedding_dimension,
    vector_io_provider_id,
):
    """Test AND filter with three or more conditions."""
    skip_if_provider_doesnt_support_filters(vector_io_provider_id)
    client = client_with_empty_registry

    vector_store = client.vector_stores.create(
        name="filter_multi_and_test",
        extra_body={
            "provider_id": vector_io_provider_id,
            "embedding_model": embedding_model_id,
        },
    )

    client.vector_io.insert(
        vector_store_id=vector_store.id,
        chunks=filter_test_chunks,
    )

    # Filter for topic == "ai" AND is_featured == True AND priority == 1
    filter_obj = CompoundFilter(
        type="and",
        filters=[
            ComparisonFilter(type="eq", key="topic", value="ai"),
            ComparisonFilter(type="eq", key="is_featured", value=True),
            ComparisonFilter(type="eq", key="priority", value=1),
        ],
    )

    response = client.vector_io.query(
        vector_store_id=vector_store.id,
        query="neural networks",
        params={"filters": filter_obj.model_dump()},
    )

    assert response is not None
    assert len(response.chunks) > 0

    # All results should match all three conditions
    for chunk in response.chunks:
        assert chunk.metadata["topic"] == "ai"
        assert chunk.metadata["is_featured"] is True
        assert chunk.metadata["priority"] == 1


# =============================================================================
# OpenAI API Filter Tests (via vector_stores.search)
# =============================================================================


@vector_provider_wrapper
def test_openai_search_with_comparison_filter(
    client_with_empty_registry,
    filter_test_chunks,
    embedding_model_id,
    embedding_dimension,
    vector_io_provider_id,
):
    """Test OpenAI-compatible search with comparison filter."""
    skip_if_provider_doesnt_support_filters(vector_io_provider_id)
    client = client_with_empty_registry

    vector_store = client.vector_stores.create(
        name="openai_filter_test",
        extra_body={
            "provider_id": vector_io_provider_id,
            "embedding_model": embedding_model_id,
        },
    )

    client.vector_io.insert(
        vector_store_id=vector_store.id,
        chunks=filter_test_chunks,
    )

    # Use the OpenAI-native `filters=` parameter (dict form) that gets parsed by openai_search_vector_store
    response = client.vector_stores.search(
        vector_store_id=vector_store.id,
        query="programming language",
        filters={"type": "eq", "key": "topic", "value": "programming"},
        max_num_results=5,
    )

    assert response is not None
    assert len(response.data) > 0

    # All results should have topic "programming"
    for result in response.data:
        assert result.attributes["topic"] == "programming"


@vector_provider_wrapper
def test_openai_search_with_compound_filter(
    client_with_empty_registry,
    filter_test_chunks,
    embedding_model_id,
    embedding_dimension,
    vector_io_provider_id,
):
    """Test OpenAI-compatible search with compound filter."""
    skip_if_provider_doesnt_support_filters(vector_io_provider_id)
    client = client_with_empty_registry

    vector_store = client.vector_stores.create(
        name="openai_compound_filter_test",
        extra_body={
            "provider_id": vector_io_provider_id,
            "embedding_model": embedding_model_id,
        },
    )

    client.vector_io.insert(
        vector_store_id=vector_store.id,
        chunks=filter_test_chunks,
    )

    # AND compound filter: category == "technology" AND is_featured == True
    response = client.vector_stores.search(
        vector_store_id=vector_store.id,
        query="artificial intelligence",
        filters={
            "type": "and",
            "filters": [
                {"type": "eq", "key": "category", "value": "technology"},
                {"type": "eq", "key": "is_featured", "value": True},
            ],
        },
        max_num_results=5,
    )

    assert response is not None
    assert len(response.data) > 0

    # All results should match both conditions
    for result in response.data:
        assert result.attributes["category"] == "technology"
        assert result.attributes["is_featured"] is True
