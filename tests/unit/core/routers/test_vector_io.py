# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from unittest.mock import AsyncMock, Mock

import pytest

from ogx.core.routers.vector_io import VectorIORouter
from ogx_api import (
    ModelNotFoundError,
    ModelType,
    ModelTypeError,
    OpenAICreateVectorStoreRequestWithExtraBody,
    OpenAISearchVectorStoreRequest,
    OpenAIUpdateVectorStoreRequest,
)


def _mock_routing_table(**kwargs):
    """Create a mock routing table with dist_registry.get_cached properly configured."""
    rt = Mock(**kwargs)
    rt.dist_registry.get_cached.return_value = Mock(provider_id="test-provider")
    return rt


async def test_single_provider_auto_selection():
    # provider_id automatically selected during vector store create() when only one provider available
    mock_routing_table = _mock_routing_table()
    mock_routing_table.impls_by_provider_id = {"inline::faiss": "mock_provider"}
    mock_routing_table.get_all_with_type = AsyncMock(
        return_value=[
            Mock(identifier="all-MiniLM-L6-v2", model_type="embedding", metadata={"embedding_dimension": 384})
        ]
    )
    mock_routing_table.get_object_by_identifier = AsyncMock(return_value=Mock(model_type=ModelType.embedding))
    mock_routing_table.register_vector_store = AsyncMock(
        return_value=Mock(identifier="vs_123", provider_id="inline::faiss", provider_resource_id="vs_123")
    )
    mock_routing_table.get_provider_impl = AsyncMock(
        return_value=Mock(openai_create_vector_store=AsyncMock(return_value=Mock(id="vs_123")))
    )
    router = VectorIORouter(mock_routing_table)
    request = OpenAICreateVectorStoreRequestWithExtraBody.model_validate(
        {"name": "test_store", "embedding_model": "all-MiniLM-L6-v2"}
    )

    result = await router.openai_create_vector_store(request)
    assert result.id == "vs_123"


async def test_create_vector_stores_multiple_providers_missing_provider_id_error():
    # if multiple providers are available, vector store create will error without provider_id
    mock_routing_table = _mock_routing_table()
    mock_routing_table.impls_by_provider_id = {
        "inline::faiss": "mock_provider_1",
        "inline::sqlite-vec": "mock_provider_2",
    }
    mock_routing_table.get_all_with_type = AsyncMock(
        return_value=[
            Mock(identifier="all-MiniLM-L6-v2", model_type="embedding", metadata={"embedding_dimension": 384})
        ]
    )
    mock_routing_table.get_object_by_identifier = AsyncMock(return_value=Mock(model_type=ModelType.embedding))
    router = VectorIORouter(mock_routing_table)
    request = OpenAICreateVectorStoreRequestWithExtraBody.model_validate(
        {"name": "test_store", "embedding_model": "all-MiniLM-L6-v2"}
    )

    with pytest.raises(ValueError, match="Multiple vector_io providers available"):
        await router.openai_create_vector_store(request)


async def test_update_vector_store_provider_id_change_fails():
    """Test that updating a vector store with a different provider_id fails with clear error."""
    mock_routing_table = _mock_routing_table()

    # Mock an existing vector store with provider_id "faiss"
    mock_existing_store = Mock()
    mock_existing_store.provider_id = "inline::faiss"
    mock_existing_store.identifier = "vs_123"

    mock_routing_table.get_object_by_identifier = AsyncMock(return_value=mock_existing_store)
    mock_routing_table.get_provider_impl = AsyncMock(
        return_value=Mock(openai_update_vector_store=AsyncMock(return_value=Mock(id="vs_123")))
    )

    router = VectorIORouter(mock_routing_table)

    # Try to update with different provider_id in metadata - this should fail
    with pytest.raises(ValueError, match="provider_id cannot be changed after vector store creation"):
        await router.openai_update_vector_store(
            vector_store_id="vs_123",
            request=OpenAIUpdateVectorStoreRequest(
                name="updated_name",
                metadata={"provider_id": "inline::sqlite"},  # Different provider_id
            ),
        )

    # Verify the existing store was looked up to check provider_id
    mock_routing_table.get_object_by_identifier.assert_called_once_with("vector_store", "vs_123")

    # Provider should not be called since validation failed
    mock_routing_table.get_provider_impl.assert_not_called()


async def test_update_vector_store_same_provider_id_succeeds():
    """Test that updating a vector store with the same provider_id succeeds."""
    mock_routing_table = _mock_routing_table()

    # Mock an existing vector store with provider_id "faiss"
    mock_existing_store = Mock()
    mock_existing_store.provider_id = "inline::faiss"
    mock_existing_store.identifier = "vs_123"

    mock_routing_table.get_object_by_identifier = AsyncMock(return_value=mock_existing_store)
    mock_routing_table.openai_update_vector_store = AsyncMock(return_value=Mock(identifier="vs_123"))

    router = VectorIORouter(mock_routing_table)

    # Update with same provider_id should succeed
    request = OpenAIUpdateVectorStoreRequest(name="updated_name", metadata={"provider_id": "inline::faiss"})
    await router.openai_update_vector_store(
        vector_store_id="vs_123",
        request=request,  # Same provider_id
    )

    # Verify the routing table method was called
    mock_routing_table.openai_update_vector_store.assert_called_once_with(vector_store_id="vs_123", request=request)


async def test_register_vector_store_only_once():
    mock_provider = Mock()
    mock_provider.register_vector_store = AsyncMock()
    mock_provider.openai_create_vector_store = AsyncMock(return_value=Mock(id="vs_123"))

    mock_routing_table = _mock_routing_table()
    mock_routing_table.impls_by_provider_id = {"inline::faiss": mock_provider}
    mock_routing_table.get_object_by_identifier = AsyncMock(return_value=Mock(model_type=ModelType.embedding))
    mock_routing_table.register_vector_store = AsyncMock(
        return_value=Mock(identifier="vs_123", provider_id="inline::faiss", provider_resource_id="vs_123")
    )
    mock_routing_table.get_provider_impl = AsyncMock(return_value=mock_provider)

    router = VectorIORouter(mock_routing_table)
    request = OpenAICreateVectorStoreRequestWithExtraBody.model_validate(
        {"name": "test_store", "embedding_model": "nomic-ai/nomic-embed-text-v1.5", "embedding_dimension": 768}
    )

    result = await router.openai_create_vector_store(request)
    assert result.id == "vs_123"

    mock_routing_table.register_vector_store.assert_called_once()
    mock_provider.openai_create_vector_store.assert_called_once()
    mock_provider.register_vector_store.assert_not_called()


async def test_create_vector_store_with_unknown_embedding_model_raises_error():
    """Test that creating a vector store with an unknown embedding model raises
    FoundError."""
    mock_routing_table = _mock_routing_table(impls_by_provider_id={"provider": "mock"})
    mock_routing_table.get_object_by_identifier = AsyncMock(return_value=None)

    router = VectorIORouter(mock_routing_table)
    request = OpenAICreateVectorStoreRequestWithExtraBody.model_validate(
        {"embedding_model": "unknown-model", "embedding_dimension": 384}
    )

    with pytest.raises(ModelNotFoundError, match="Model 'unknown-model' not found"):
        await router.openai_create_vector_store(request)


async def test_create_vector_store_with_wrong_model_type_raises_error():
    """Test that creating a vector store with a non-embedding model raises ModelTypeError."""
    mock_routing_table = _mock_routing_table(impls_by_provider_id={"provider": "mock"})
    mock_routing_table.get_object_by_identifier = AsyncMock(return_value=Mock(model_type=ModelType.llm))

    router = VectorIORouter(mock_routing_table)
    request = OpenAICreateVectorStoreRequestWithExtraBody.model_validate(
        {"embedding_model": "text-model", "embedding_dimension": 384}
    )

    with pytest.raises(ModelTypeError, match="Model 'text-model' is of type"):
        await router.openai_create_vector_store(request)


async def test_query_rewrite_functionality():
    """Test query rewriting at the router level."""
    from unittest.mock import MagicMock

    from ogx.core.datatypes import QualifiedModel, RewriteQueryParams, VectorStoresConfig
    from ogx_api import VectorStoreSearchResponsePage

    mock_routing_table = _mock_routing_table()

    # Mock routing table method that returns search results
    mock_search_response = VectorStoreSearchResponsePage(search_query=["rewritten test query"], data=[], has_more=False)
    mock_routing_table.openai_search_vector_store = AsyncMock(return_value=mock_search_response)

    # Mock inference API for query rewriting
    mock_inference_api = Mock()
    mock_inference_api.openai_chat_completion = AsyncMock(
        return_value=MagicMock(choices=[MagicMock(message=MagicMock(content="rewritten test query"))])
    )

    # Create config with rewrite params
    vector_stores_config = VectorStoresConfig(
        rewrite_query_params=RewriteQueryParams(
            model=QualifiedModel(provider_id="test", model_id="llama"),
            max_tokens=100,
            temperature=0.3,
        )
    )

    router = VectorIORouter(mock_routing_table, vector_stores_config, mock_inference_api)

    # Test query rewrite with rewrite_query=True
    request = OpenAISearchVectorStoreRequest(
        query="test query",
        rewrite_query=True,
        max_num_results=5,
    )
    result = await router.openai_search_vector_store(
        vector_store_id="vs_123",
        request=request,
    )

    # Verify chat completion was called for query rewriting
    assert mock_inference_api.openai_chat_completion.called
    chat_call_args = mock_inference_api.openai_chat_completion.call_args[0][0]
    assert chat_call_args.model == "test/llama"

    # Verify default prompt is used
    prompt_text = chat_call_args.messages[0].content
    expected_prompt = "Expand this query with relevant synonyms and related terms. Return only the improved query, no explanations:\n\ntest query\n\nImproved query:"
    assert prompt_text == expected_prompt

    # Verify routing table was called with rewritten query and rewrite_query=False
    mock_routing_table.openai_search_vector_store.assert_called_once()
    call_kwargs = mock_routing_table.openai_search_vector_store.call_args.kwargs
    assert call_kwargs["vector_store_id"] == "vs_123"
    forward_request = call_kwargs["request"]
    assert forward_request.query == "rewritten test query"
    assert forward_request.rewrite_query is False  # Should be False since router handled it

    assert result is not None


async def test_query_rewrite_error_when_not_configured():
    """Test that query rewriting fails with proper error when not configured."""
    mock_routing_table = _mock_routing_table()
    mock_provider = Mock()
    mock_routing_table.get_provider_impl = AsyncMock(return_value=mock_provider)

    # No config or inference API
    router = VectorIORouter(mock_routing_table)

    with pytest.raises(ValueError, match="Query rewriting is not available"):
        await router.openai_search_vector_store(
            vector_store_id="vs_123",
            request=OpenAISearchVectorStoreRequest(
                query="test query",
                rewrite_query=True,
                max_num_results=5,
            ),
        )


async def test_query_rewrite_with_custom_prompt():
    """Test query rewriting with custom prompt."""
    from unittest.mock import MagicMock

    from ogx.core.datatypes import QualifiedModel, RewriteQueryParams, VectorStoresConfig
    from ogx_api import VectorStoreSearchResponsePage

    mock_routing_table = _mock_routing_table()

    mock_search_response = VectorStoreSearchResponsePage(search_query=["custom rewrite"], data=[], has_more=False)
    mock_routing_table.openai_search_vector_store = AsyncMock(return_value=mock_search_response)

    mock_inference_api = Mock()
    mock_inference_api.openai_chat_completion = AsyncMock(
        return_value=MagicMock(choices=[MagicMock(message=MagicMock(content="custom rewrite"))])
    )

    vector_stores_config = VectorStoresConfig(
        rewrite_query_params=RewriteQueryParams(
            model=QualifiedModel(provider_id="test", model_id="llama"),
            prompt="Custom prompt: {query}",
            max_tokens=150,
            temperature=0.7,
        )
    )

    router = VectorIORouter(mock_routing_table, vector_stores_config, mock_inference_api)

    await router.openai_search_vector_store(
        vector_store_id="vs_123",
        request=OpenAISearchVectorStoreRequest(
            query="test query",
            rewrite_query=True,
            max_num_results=5,
        ),
    )

    # Verify custom prompt was used
    chat_call_args = mock_inference_api.openai_chat_completion.call_args[0][0]
    assert chat_call_args.messages[0].content == "Custom prompt: test query"
    assert chat_call_args.max_tokens == 150
    assert chat_call_args.temperature == 0.7


async def test_search_without_rewrite():
    """Test that search without rewrite_query doesn't call inference API."""
    from ogx_api import VectorStoreSearchResponsePage

    mock_routing_table = _mock_routing_table()

    mock_search_response = VectorStoreSearchResponsePage(search_query=["test query"], data=[], has_more=False)
    mock_routing_table.openai_search_vector_store = AsyncMock(return_value=mock_search_response)

    mock_inference_api = Mock()
    mock_inference_api.openai_chat_completion = AsyncMock()

    router = VectorIORouter(mock_routing_table, inference_api=mock_inference_api)

    await router.openai_search_vector_store(
        vector_store_id="vs_123",
        request=OpenAISearchVectorStoreRequest(
            query="test query",
            rewrite_query=False,
            max_num_results=5,
        ),
    )

    # Verify inference API was NOT called
    assert not mock_inference_api.openai_chat_completion.called

    # Verify routing table was called with original query
    mock_routing_table.openai_search_vector_store.assert_called_once()
    call_kwargs = mock_routing_table.openai_search_vector_store.call_args.kwargs
    assert call_kwargs["vector_store_id"] == "vs_123"
    forward_request = call_kwargs["request"]
    assert forward_request.query == "test query"
