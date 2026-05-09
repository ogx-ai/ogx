# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import time

import pytest

from ogx_api import ChunkMetadata, EmbeddedChunk

from ..conftest import vector_provider_wrapper


@pytest.fixture(scope="session")
def sample_chunks(embedding_dimension):
    import numpy as np

    from ogx.providers.utils.vector_io.vector_utils import generate_chunk_id

    chunks_data = [
        (
            "Python is a high-level programming language that emphasizes code readability and allows programmers to express concepts in fewer lines of code than would be possible in languages such as C++ or Java.",
            "doc1",
        ),
        (
            "Machine learning is a subset of artificial intelligence that enables systems to automatically learn and improve from experience without being explicitly programmed, using statistical techniques to give computer systems the ability to progressively improve performance on a specific task.",
            "doc2",
        ),
        (
            "Data structures are fundamental to computer science because they provide organized ways to store and access data efficiently, enable faster processing of data through optimized algorithms, and form the building blocks for more complex software systems.",
            "doc3",
        ),
        (
            "Neural networks are inspired by biological neural networks found in animal brains, using interconnected nodes called artificial neurons to process information through weighted connections that can be trained to recognize patterns and solve complex problems through iterative learning.",
            "doc4",
        ),
    ]

    # Use a fixed seed for deterministic embeddings
    np.random.seed(42)

    embedded_chunks = []
    for _i, (content, doc_id) in enumerate(chunks_data):
        chunk_id = generate_chunk_id(doc_id, content)
        embedding = np.random.random(int(embedding_dimension)).tolist()  # Generate deterministic dummy embeddings
        embedded_chunk = EmbeddedChunk(
            content=content,
            chunk_id=chunk_id,
            metadata={"document_id": doc_id},
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
    def clear_registry():
        vector_stores = client_with_models.vector_stores.list()
        for vector_store in vector_stores.data:
            client_with_models.vector_stores.delete(vector_store_id=vector_store.id)

    clear_registry()
    yield client_with_models

    # you must clean after the last test if you were running tests against
    # a stateful server instance
    clear_registry()


@vector_provider_wrapper
def test_vector_store_retrieve(
    client_with_empty_registry, embedding_model_id, embedding_dimension, vector_io_provider_id
):
    vector_store_name = "test_vector_store"
    create_response = client_with_empty_registry.vector_stores.create(
        name=vector_store_name,
        extra_body={
            "provider_id": vector_io_provider_id,
            "embedding_model": embedding_model_id,
        },
    )

    actual_vector_store_id = create_response.id

    # Retrieve the vector store and validate its properties
    response = client_with_empty_registry.vector_stores.retrieve(vector_store_id=actual_vector_store_id)
    assert response is not None
    assert response.id == actual_vector_store_id
    assert response.name == vector_store_name
    assert response.id.startswith("vs_")


@vector_provider_wrapper
def test_vector_store_register(
    client_with_empty_registry, embedding_model_id, embedding_dimension, vector_io_provider_id
):
    vector_store_name = "test_vector_store"
    response = client_with_empty_registry.vector_stores.create(
        name=vector_store_name,
        extra_body={
            "provider_id": vector_io_provider_id,
            "embedding_model": embedding_model_id,
        },
    )

    actual_vector_store_id = response.id
    assert actual_vector_store_id.startswith("vs_")
    assert actual_vector_store_id != vector_store_name

    vector_stores = client_with_empty_registry.vector_stores.list()
    assert len(vector_stores.data) == 1
    vector_store = vector_stores.data[0]
    assert vector_store.id == actual_vector_store_id
    assert vector_store.name == vector_store_name

    client_with_empty_registry.vector_stores.delete(vector_store_id=actual_vector_store_id)

    vector_stores = client_with_empty_registry.vector_stores.list()
    assert len(vector_stores.data) == 0


@pytest.mark.parametrize(
    "test_case",
    [
        ("What makes Python different from C++ and Java?", "doc1"),
        ("How do systems learn without explicit programming?", "doc2"),
        ("Why are data structures important in computer science?", "doc3"),
        ("What is the biological inspiration for neural networks?", "doc4"),
        ("How does machine learning improve over time?", "doc2"),
    ],
)
@vector_provider_wrapper
def test_insert_chunks(
    client_with_empty_registry, embedding_model_id, embedding_dimension, sample_chunks, test_case, vector_io_provider_id
):
    vector_store_name = "test_vector_store"
    create_response = client_with_empty_registry.vector_stores.create(
        name=vector_store_name,
        extra_body={
            "provider_id": vector_io_provider_id,
        },
    )

    actual_vector_store_id = create_response.id

    client_with_empty_registry.vector_io.insert(
        vector_store_id=actual_vector_store_id,
        chunks=sample_chunks,
    )

    response = client_with_empty_registry.vector_io.query(
        vector_store_id=actual_vector_store_id,
        query="What is the capital of France?",
    )
    assert response is not None
    assert len(response.chunks) > 1
    assert len(response.scores) > 1

    query, expected_doc_id = test_case
    response = client_with_empty_registry.vector_io.query(
        vector_store_id=actual_vector_store_id,
        query=query,
    )
    assert response is not None
    top_match = response.chunks[0]
    assert top_match is not None
    # Note: Using dummy embeddings, so we can't test semantic accuracy
    # Just verify that the system returns results and has correct structure
    assert "document_id" in top_match.metadata
    assert top_match.metadata["document_id"] in ["doc1", "doc2", "doc3", "doc4"]


@vector_provider_wrapper
def test_insert_chunks_with_precomputed_embeddings(
    client_with_empty_registry, embedding_model_id, embedding_dimension, vector_io_provider_id
):
    vector_io_provider_params_dict = {
        "inline::milvus": {"score_threshold": -1.0},
        "inline::qdrant": {"score_threshold": -1.0},
        "remote::qdrant": {"score_threshold": -1.0},
        "remote::elasticsearch": {"score_threshold": -1.0},
    }
    vector_store_name = "test_precomputed_embeddings_db"
    register_response = client_with_empty_registry.vector_stores.create(
        name=vector_store_name,
        extra_body={
            "provider_id": vector_io_provider_id,
        },
    )

    actual_vector_store_id = register_response.id

    chunks_with_embeddings = [
        EmbeddedChunk(
            content="This is a test chunk with precomputed embedding.",
            chunk_id="chunk1",
            metadata={"document_id": "doc1", "source": "precomputed", "chunk_id": "chunk1"},
            chunk_metadata=ChunkMetadata(
                document_id="doc1",
                chunk_id="chunk1",
                source="precomputed",
                created_timestamp=int(time.time()),
                updated_timestamp=int(time.time()),
                content_token_count=8,
            ),
            embedding=[0.1] * int(embedding_dimension),
            embedding_model="test-embedding-model",
            embedding_dimension=int(embedding_dimension),
        ),
    ]

    client_with_empty_registry.vector_io.insert(
        vector_store_id=actual_vector_store_id,
        chunks=chunks_with_embeddings,
    )

    provider = [p.provider_id for p in client_with_empty_registry.providers.list() if p.api == "vector_io"][0]
    response = client_with_empty_registry.vector_io.query(
        vector_store_id=actual_vector_store_id,
        query="precomputed embedding test",
        params=vector_io_provider_params_dict.get(provider, None),
    )

    # Verify the top result is the expected document
    assert response is not None
    assert len(response.chunks) > 0, (
        f"provider params for {provider} = {vector_io_provider_params_dict.get(provider, None)}"
    )
    assert response.chunks[0].metadata["document_id"] == "doc1"
    assert response.chunks[0].metadata["source"] == "precomputed"


# expect this test to fail
@vector_provider_wrapper
def test_query_returns_valid_object_when_identical_to_embedding_in_vdb(
    client_with_empty_registry, embedding_model_id, embedding_dimension, vector_io_provider_id
):
    vector_io_provider_params_dict = {
        "inline::milvus": {"score_threshold": 0.0},
        "remote::qdrant": {"score_threshold": 0.0},
        "inline::qdrant": {"score_threshold": 0.0},
        "remote::elasticsearch": {"score_threshold": 0.0},
    }
    vector_store_name = "test_precomputed_embeddings_db"
    register_response = client_with_empty_registry.vector_stores.create(
        name=vector_store_name,
        extra_body={
            "embedding_model": embedding_model_id,
            "provider_id": vector_io_provider_id,
        },
    )

    actual_vector_store_id = register_response.id

    from ogx.providers.utils.vector_io.vector_utils import generate_chunk_id

    chunk_id = generate_chunk_id("doc1", "duplicate")
    chunks_with_embeddings = [
        EmbeddedChunk(
            content="duplicate",
            chunk_id=chunk_id,
            metadata={"document_id": "doc1", "source": "precomputed"},
            chunk_metadata=ChunkMetadata(
                document_id="doc1",
                chunk_id=chunk_id,
                source="precomputed",
                created_timestamp=int(time.time()),
                updated_timestamp=int(time.time()),
                content_token_count=1,
            ),
            embedding=[0.1] * int(embedding_dimension),
            embedding_model="test-embedding-model",
            embedding_dimension=int(embedding_dimension),
        ),
    ]

    client_with_empty_registry.vector_io.insert(
        vector_store_id=actual_vector_store_id,
        chunks=chunks_with_embeddings,
    )

    provider = [p.provider_id for p in client_with_empty_registry.providers.list() if p.api == "vector_io"][0]
    response = client_with_empty_registry.vector_io.query(
        vector_store_id=actual_vector_store_id,
        query="duplicate",
        params=vector_io_provider_params_dict.get(provider, None),
    )

    # Verify the top result is the expected document
    assert response is not None
    assert len(response.chunks) > 0
    assert response.chunks[0].metadata["document_id"] == "doc1"
    assert response.chunks[0].metadata["source"] == "precomputed"


@vector_provider_wrapper
def test_auto_extract_embedding_dimension(
    client_with_empty_registry, embedding_model_id, embedding_dimension, vector_io_provider_id
):
    # This test specifically tests embedding model override, so we keep embedding_model
    vs = client_with_empty_registry.vector_stores.create(
        name="test_auto_extract",
        extra_body={"embedding_model": embedding_model_id, "provider_id": vector_io_provider_id},
    )
    assert vs.id is not None


@vector_provider_wrapper
def test_provider_auto_selection_single_provider(
    client_with_empty_registry, embedding_model_id, embedding_dimension, vector_io_provider_id
):
    providers = [p for p in client_with_empty_registry.providers.list() if p.api == "vector_io"]
    if len(providers) != 1:
        pytest.skip(f"Test requires exactly one vector_io provider, found {len(providers)}")

    # Test that when only one provider is available, it's auto-selected (no provider_id needed)
    vs = client_with_empty_registry.vector_stores.create(name="test_auto_provider")
    assert vs.id is not None


@vector_provider_wrapper
def test_provider_id_override(
    client_with_empty_registry, embedding_model_id, embedding_dimension, vector_io_provider_id
):
    providers = [p for p in client_with_empty_registry.providers.list() if p.api == "vector_io"]
    if len(providers) != 1:
        pytest.skip(f"Test requires exactly one vector_io provider, found {len(providers)}")

    provider_id = providers[0].provider_id

    # Test explicit provider_id specification (using default embedding model)
    vs = client_with_empty_registry.vector_stores.create(
        name="test_provider_override", extra_body={"provider_id": provider_id}
    )
    assert vs.id is not None
    assert vs.metadata.get("provider_id") == provider_id
