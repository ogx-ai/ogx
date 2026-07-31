# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from ogx.core.datatypes import ContextualRetrievalParams, QualifiedModel, VectorStoresConfig
from ogx.providers.utils.memory.openai_vector_store_mixin import OpenAIVectorStoreMixin
from ogx_api import (
    Chunk,
    ChunkMetadata,
    DeleteChunksRequest,
    InsertChunksRequest,
    OpenAIChatCompletion,
    QueryChunksRequest,
    QueryChunksResponse,
    TextContentItem,
    VectorStoreChunkingStrategyContextualConfig,
)

# This test is a unit test for contextual retrieval functionality in the OpenAIVectorStoreMixin.
# More general (API-level) tests should be placed in tests/integration/vector_io/
#
# How to run this test:
#
# pytest tests/unit/providers/vector_io/test_contextual_retrieval.py \
# -v -s --tb=short --disable-warnings


class MockInferenceAPI:
    def __init__(self):
        self.openai_chat_completion = AsyncMock()


class MockVectorStoreProvider(OpenAIVectorStoreMixin):
    def __init__(self, inference_api):
        super().__init__(inference_api=inference_api)
        self.vector_stores_config = VectorStoresConfig(contextual_retrieval_params=ContextualRetrievalParams())

    async def register_vector_store(self, vector_store) -> None:
        pass

    async def unregister_vector_store(self, vector_store_id) -> None:
        pass

    async def insert_chunks(self, request: InsertChunksRequest) -> None:
        pass

    async def query_chunks(self, request: QueryChunksRequest) -> QueryChunksResponse:
        return QueryChunksResponse(chunks=[], scores=[])

    async def delete_chunks(self, request: DeleteChunksRequest) -> None:
        pass


@pytest.fixture
def inference_api():
    """Fixture providing a mock inference API."""
    return MockInferenceAPI()


@pytest.fixture
def provider(inference_api):
    """Fixture providing a mock vector store provider with contextual retrieval support."""
    return MockVectorStoreProvider(inference_api)


@pytest.fixture
def strategy_config():
    """Fixture providing a default contextual chunking strategy configuration."""
    return VectorStoreChunkingStrategyContextualConfig(
        model_id="llama3.2", max_chunk_size_tokens=700, chunk_overlap_tokens=400
    )


@pytest.fixture
def provider_with_model(inference_api):
    """Fixture providing a provider with model configured in contextual_retrieval_params."""
    provider = MockVectorStoreProvider(inference_api)
    provider.vector_stores_config = VectorStoresConfig(
        contextual_retrieval_params=ContextualRetrievalParams(
            model=QualifiedModel(provider_id="meta", model_id="llama3.2")
        )
    )
    return provider


@pytest.fixture(autouse=False)
def fast_retry(monkeypatch):
    """Eliminate backoff delays in retry tests."""
    import ogx.providers.utils.memory.openai_vector_store_mixin as mod

    monkeypatch.setattr(mod, "_RETRY_BASE_DELAY", 0.0)
    monkeypatch.setattr(mod.random, "uniform", lambda _a, _b: 0.0)


def create_mock_response(content):
    """Create a mock OpenAI chat completion response with the given content."""
    mock_response = MagicMock(spec=OpenAIChatCompletion)
    mock_message = MagicMock()
    mock_message.content = content
    mock_choice = MagicMock()
    mock_choice.message = mock_message
    mock_response.choices = [mock_choice]
    return mock_response


class _RetriableError(Exception):
    """Exception with a response attribute for retry-logic tests."""

    def __init__(self, message: str, status_code: int):
        super().__init__(message)
        self.response = MagicMock(status_code=status_code)


def create_chunk(content, chunk_id="chunk_1", document_id="doc_1"):
    """Create a chunk with the given content and IDs."""
    return Chunk(
        content=content, chunk_id=chunk_id, chunk_metadata=ChunkMetadata(chunk_id=chunk_id, document_id=document_id)
    )


async def test_execute_contextual_chunk_transformation_success(inference_api, provider, strategy_config):
    """Test that contextual retrieval successfully prepends context to chunk content."""
    inference_api.openai_chat_completion.return_value = create_mock_response("Situational context")

    chunks = [create_chunk("Original content")]

    await provider._execute_contextual_chunk_transformation(chunks, "Whole document", strategy_config)

    assert chunks[0].content == "Situational context\n\nOriginal content"
    inference_api.openai_chat_completion.assert_called_once()


async def test_execute_contextual_chunk_transformation_list_content(inference_api, provider, strategy_config):
    """Test that list content (TextContentItem) is handled correctly."""
    inference_api.openai_chat_completion.return_value = create_mock_response("Situational context")

    chunks = [
        Chunk(
            content=[TextContentItem(text="Original content")],
            chunk_id="chunk_1",
            chunk_metadata=ChunkMetadata(chunk_id="chunk_1", document_id="doc_1"),
        )
    ]

    await provider._execute_contextual_chunk_transformation(chunks, "Whole document", strategy_config)

    content = chunks[0].content
    assert isinstance(content, list)
    assert len(content) == 2
    item0 = content[0]
    item1 = content[1]
    assert isinstance(item0, TextContentItem)
    assert isinstance(item1, TextContentItem)
    assert item0.text == "Situational context\n\n"
    assert item1.text == "Original content"


async def test_execute_contextual_chunk_transformation_partial_failure(inference_api, provider, strategy_config):
    """Test that partial failures are handled gracefully (successful chunks are updated)."""
    inference_api.openai_chat_completion.side_effect = [
        create_mock_response("Context 1"),
        Exception("Inference failed"),
    ]

    chunks = [
        create_chunk("C1", chunk_id="1", document_id="d"),
        create_chunk("C2", chunk_id="2", document_id="d"),
    ]

    await provider._execute_contextual_chunk_transformation(chunks, "Doc", strategy_config)

    assert chunks[0].content == "Context 1\n\nC1"
    assert chunks[1].content == "C2"  # Unchanged


async def test_execute_contextual_chunk_transformation_total_failure(inference_api, provider, strategy_config):
    """Test that total failure raises RuntimeError."""
    inference_api.openai_chat_completion.side_effect = Exception("Total failure")

    chunks = [create_chunk("C1", chunk_id="1", document_id="d")]

    with pytest.raises(RuntimeError, match="Failed to contextualize any chunks"):
        await provider._execute_contextual_chunk_transformation(chunks, "Doc", strategy_config)


async def test_execute_contextual_chunk_transformation_custom_prompt(inference_api, provider):
    """Test that custom prompts are used correctly."""
    inference_api.openai_chat_completion.return_value = create_mock_response("Context")

    chunks = [create_chunk("C1", chunk_id="1", document_id="d")]
    custom_prompt = "Custom: {{WHOLE_DOCUMENT}} + {{CHUNK_CONTENT}}"
    config = VectorStoreChunkingStrategyContextualConfig(
        model_id="llama3.2", context_prompt=custom_prompt, max_chunk_size_tokens=700, chunk_overlap_tokens=400
    )

    await provider._execute_contextual_chunk_transformation(chunks, "DOC_BODY", config)

    args, _ = inference_api.openai_chat_completion.call_args
    messages = args[0].messages
    # Document placed in system message for prefix caching, chunk in user message
    assert len(messages) == 2
    assert messages[0].role == "system"
    assert messages[0].content == "Custom: DOC_BODY +"
    assert messages[1].role == "user"
    assert messages[1].content == "C1"


async def test_execute_contextual_chunk_transformation_empty_response(inference_api, provider, strategy_config):
    """Test that empty choices are handled gracefully (log error and count as failure)."""
    mock_response = MagicMock(spec=OpenAIChatCompletion)
    mock_response.choices = []
    inference_api.openai_chat_completion.return_value = mock_response

    chunks = [create_chunk("C1", chunk_id="1", document_id="d")]

    with pytest.raises(RuntimeError, match="Failed to contextualize any chunks"):
        await provider._execute_contextual_chunk_transformation(chunks, "Doc", strategy_config)


async def test_execute_contextual_chunk_transformation_timeout(inference_api, provider, fast_retry):
    """Test that timeout errors are handled correctly."""

    async def slow_llm(*_args, **_kwargs):
        await asyncio.sleep(5)

    inference_api.openai_chat_completion.side_effect = slow_llm

    chunks = [create_chunk("C1", chunk_id="1", document_id="d")]
    config = VectorStoreChunkingStrategyContextualConfig(model_id="llama3.2", timeout_seconds=1)

    with pytest.raises(RuntimeError, match="Failed to contextualize any chunks"):
        await provider._execute_contextual_chunk_transformation(chunks, "Doc", config)


async def test_execute_contextual_chunk_transformation_all_empty_context_raises(inference_api, provider):
    """Test that all chunks returning empty context raises RuntimeError."""
    inference_api.openai_chat_completion.return_value = create_mock_response("")

    chunks = [create_chunk("Original", chunk_id="1", document_id="d")]
    config = VectorStoreChunkingStrategyContextualConfig(model_id="llama3.2")

    with pytest.raises(RuntimeError, match="Failed to generate context using model"):
        await provider._execute_contextual_chunk_transformation(chunks, "Doc", config)

    assert chunks[0].content == "Original"


async def test_execute_contextual_chunk_transformation_partial_empty_context(inference_api, provider):
    """Test that partial empty context logs warning but succeeds."""
    inference_api.openai_chat_completion.side_effect = [
        create_mock_response("Context 1"),
        create_mock_response(""),
    ]

    chunks = [
        create_chunk("C1", chunk_id="1", document_id="d"),
        create_chunk("C2", chunk_id="2", document_id="d"),
    ]
    config = VectorStoreChunkingStrategyContextualConfig(model_id="llama3.2")

    await provider._execute_contextual_chunk_transformation(chunks, "Doc", config)

    assert chunks[0].content == "Context 1\n\nC1"
    assert chunks[1].content == "C2"  # Unchanged


def test_contextual_config_validation_model_id_required():
    """Test that model_id is required and cannot be empty."""
    with pytest.raises(ValueError):
        VectorStoreChunkingStrategyContextualConfig(model_id="")


def test_contextual_config_validation_overlap_less_than_size():
    """Test that chunk_overlap_tokens must be less than max_chunk_size_tokens."""
    with pytest.raises(ValueError, match="chunk_overlap_tokens must be less than max_chunk_size_tokens"):
        VectorStoreChunkingStrategyContextualConfig(
            model_id="llama3.2",
            max_chunk_size_tokens=500,
            chunk_overlap_tokens=500,
        )


def test_contextual_config_validation_prompt_placeholders():
    """Test that custom prompts must contain required placeholders."""
    with pytest.raises(ValueError, match="must contain"):
        VectorStoreChunkingStrategyContextualConfig(
            model_id="llama3.2",
            context_prompt="Missing placeholders",
        )


def test_contextual_config_validation_prompt_ordering():
    """Test that {{WHOLE_DOCUMENT}} must appear before {{CHUNK_CONTENT}} in context_prompt."""
    with pytest.raises(ValueError, match="must have.*WHOLE_DOCUMENT.*before.*CHUNK_CONTENT"):
        VectorStoreChunkingStrategyContextualConfig(
            model_id="llama3.2",
            context_prompt="{{CHUNK_CONTENT}} then {{WHOLE_DOCUMENT}}",
        )


async def test_execute_contextual_chunk_transformation_document_too_large(provider, strategy_config):
    """Test that documents exceeding max_document_tokens raise a clear error."""
    max_tokens = provider.vector_stores_config.contextual_retrieval_params.max_document_tokens
    huge_content = "x" * (max_tokens * 4 + 100)
    chunks = [create_chunk("Small chunk")]

    with pytest.raises(ValueError, match="Failed to process document for contextual retrieval"):
        await provider._execute_contextual_chunk_transformation(chunks, huge_content, strategy_config)


async def test_execute_contextual_chunk_transformation_model_fallback(inference_api, provider_with_model):
    """Test that model_id falls back to contextual_retrieval_params.model."""
    inference_api.openai_chat_completion.return_value = create_mock_response("Context")
    chunks = [create_chunk("C1")]
    config = VectorStoreChunkingStrategyContextualConfig()  # No model_id

    await provider_with_model._execute_contextual_chunk_transformation(chunks, "Doc", config)

    args, _ = inference_api.openai_chat_completion.call_args
    assert args[0].model == "meta/llama3.2"


async def test_execute_contextual_chunk_transformation_retries_on_retriable_error(inference_api, provider, fast_retry):
    """Test that retriable errors (e.g. rate limits) are retried with backoff."""
    inference_api.openai_chat_completion.side_effect = [
        _RetriableError("Rate limit exceeded", 429),
        create_mock_response("Context after retry"),
    ]

    chunks = [create_chunk("C1", chunk_id="1", document_id="d")]
    config = VectorStoreChunkingStrategyContextualConfig(model_id="llama3.2")

    await provider._execute_contextual_chunk_transformation(chunks, "Doc", config)

    assert chunks[0].content == "Context after retry\n\nC1"
    assert inference_api.openai_chat_completion.call_count == 2


async def test_execute_contextual_chunk_transformation_no_retry_on_non_retriable(inference_api, provider):
    """Test that non-retriable errors fail immediately without retry."""
    inference_api.openai_chat_completion.side_effect = ValueError("Bad request")

    chunks = [create_chunk("C1", chunk_id="1", document_id="d")]
    config = VectorStoreChunkingStrategyContextualConfig(model_id="llama3.2")

    with pytest.raises(RuntimeError, match="Failed to contextualize any chunks"):
        await provider._execute_contextual_chunk_transformation(chunks, "Doc", config)

    assert inference_api.openai_chat_completion.call_count == 1


async def test_execute_contextual_chunk_transformation_retry_exhaustion(inference_api, provider, fast_retry):
    """Test that exhausted retries result in failure."""
    inference_api.openai_chat_completion.side_effect = [_RetriableError("Rate limit exceeded", 429)] * 5

    chunks = [create_chunk("C1", chunk_id="1", document_id="d")]
    config = VectorStoreChunkingStrategyContextualConfig(model_id="llama3.2")

    with pytest.raises(RuntimeError, match="Failed to contextualize any chunks"):
        await provider._execute_contextual_chunk_transformation(chunks, "Doc", config)

    # 1 initial + 3 retries = 4 attempts
    assert inference_api.openai_chat_completion.call_count == 4
