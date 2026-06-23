# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from unittest.mock import AsyncMock

from ogx.providers.inline.responses.builtin.config import MemoryConfig
from ogx.providers.inline.responses.builtin.responses.memory import (
    build_memory_filters,
    extract_memory_query,
    resolve_memory_context,
)
from ogx_api import OpenAIResponseInputMessageContentText, OpenAIResponseMessage
from ogx_api.responses.models import MemoryToolConfig
from ogx_api.vector_io.models import VectorStoreContent, VectorStoreSearchResponse, VectorStoreSearchResponsePage
from tests.unit.providers.responses.builtin.test_openai_responses_helpers import fake_stream


def test_extract_memory_query_uses_string_input():
    query = extract_memory_query("remember my repo preferences")

    assert query == "remember my repo preferences"


def test_extract_memory_query_uses_latest_user_text():
    query = extract_memory_query(
        [
            OpenAIResponseMessage(role="user", content="first turn"),
            OpenAIResponseMessage(role="assistant", content="assistant text"),
            OpenAIResponseMessage(
                role="user",
                content=[OpenAIResponseInputMessageContentText(text="latest user turn")],
            ),
        ]
    )

    assert query == "latest user turn"


def test_build_memory_filters_requires_owner_scope():
    filters = build_memory_filters(
        memory_config=MemoryConfig(),
        owner_id="user-123",
        request_filters={"type": "eq", "key": "project", "value": "ogx"},
    )

    assert filters == {
        "type": "and",
        "filters": [
            {"type": "eq", "key": "memory", "value": True},
            {"type": "eq", "key": "owner_id", "value": "user-123"},
            {"type": "eq", "key": "project", "value": "ogx"},
        ],
    }


def test_build_memory_filters_omits_owner_when_not_available():
    filters = build_memory_filters(
        memory_config=MemoryConfig(),
        owner_id=None,
        request_filters={"type": "eq", "key": "project", "value": "ogx"},
    )

    assert filters == {
        "type": "and",
        "filters": [
            {"type": "eq", "key": "memory", "value": True},
            {"type": "eq", "key": "project", "value": "ogx"},
        ],
    }


async def test_resolve_memory_context_searches_with_owner_filter():
    vector_io = AsyncMock()
    vector_io.openai_search_vector_store.return_value = VectorStoreSearchResponsePage(
        search_query=["repo prefs"],
        has_more=False,
        data=[
            VectorStoreSearchResponse(
                file_id="file_1",
                filename="memory.md",
                score=0.9,
                attributes={"owner_id": "user-123", "memory": True, "created_at": 123.0},
                content=[VectorStoreContent(type="text", text="Prefers small stacked PRs.")],
            )
        ],
    )

    context = await resolve_memory_context(
        vector_io_api=vector_io,
        memory_config=MemoryConfig(default_vector_store_id="vs_mem"),
        request_memory=MemoryToolConfig(owner_id="user-123"),
        input="repo prefs",
        metadata=None,
        safety_identifier=None,
    )

    assert context is not None
    assert "Prefers small stacked PRs." in context
    request = vector_io.openai_search_vector_store.call_args.kwargs["request"]
    assert request.filters["filters"][1] == {"type": "eq", "key": "owner_id", "value": "user-123"}


async def test_resolve_memory_context_skips_without_owner_when_required():
    vector_io = AsyncMock()

    context = await resolve_memory_context(
        vector_io_api=vector_io,
        memory_config=MemoryConfig(default_vector_store_id="vs_mem", require_owner=True),
        request_memory=MemoryToolConfig(),
        input="repo prefs",
        metadata=None,
        safety_identifier=None,
    )

    assert context is None
    vector_io.openai_search_vector_store.assert_not_called()


async def test_resolve_memory_context_searches_without_owner_when_optional():
    vector_io = AsyncMock()
    vector_io.openai_search_vector_store.return_value = VectorStoreSearchResponsePage(
        search_query=["repo prefs"],
        has_more=False,
        data=[
            VectorStoreSearchResponse(
                file_id="file_1",
                filename="memory.md",
                score=0.9,
                attributes={"memory": True, "created_at": 123.0},
                content=[VectorStoreContent(type="text", text="Global rollout preference.")],
            )
        ],
    )

    context = await resolve_memory_context(
        vector_io_api=vector_io,
        memory_config=MemoryConfig(default_vector_store_id="vs_mem", require_owner=False),
        request_memory=MemoryToolConfig(),
        input="repo prefs",
        metadata=None,
        safety_identifier=None,
    )

    assert context is not None
    assert "Global rollout preference." in context
    request = vector_io.openai_search_vector_store.call_args.kwargs["request"]
    assert request.filters == {"type": "and", "filters": [{"type": "eq", "key": "memory", "value": True}]}


async def test_resolve_memory_context_continues_on_search_error():
    vector_io = AsyncMock()
    vector_io.openai_search_vector_store.side_effect = RuntimeError("boom")

    context = await resolve_memory_context(
        vector_io_api=vector_io,
        memory_config=MemoryConfig(default_vector_store_id="vs_mem"),
        request_memory=MemoryToolConfig(owner_id="user-123"),
        input="repo prefs",
        metadata=None,
        safety_identifier=None,
    )

    assert context is None


async def test_memory_context_is_injected_without_file_search_output(
    openai_responses_impl,
    mock_inference_api,
    mock_vector_io_api,
):
    mock_inference_api.openai_chat_completion.return_value = fake_stream()
    mock_vector_io_api.openai_search_vector_store.return_value = VectorStoreSearchResponsePage(
        search_query=["repo prefs"],
        has_more=False,
        data=[
            VectorStoreSearchResponse(
                file_id="file_1",
                filename="memory.md",
                score=0.9,
                attributes={"owner_id": "user-123", "memory": True},
                content=[VectorStoreContent(type="text", text="Prefers signed-off commits.")],
            )
        ],
    )
    openai_responses_impl.memory_config.default_vector_store_id = "vs_mem"

    result = await openai_responses_impl.create_openai_response(
        input="repo prefs",
        model="test-model",
        stream=True,
        memory=MemoryToolConfig(owner_id="user-123"),
    )
    chunks = [chunk async for chunk in result]

    request = mock_inference_api.openai_chat_completion.call_args.args[0]
    assert any("Prefers signed-off commits." in message.content for message in request.messages)
    assert all(getattr(chunk, "item", None) is None or chunk.item.type != "file_search_call" for chunk in chunks)


async def test_memory_disabled_does_not_search(
    openai_responses_impl,
    mock_inference_api,
    mock_vector_io_api,
):
    mock_inference_api.openai_chat_completion.return_value = fake_stream()
    openai_responses_impl.memory_config.default_vector_store_id = "vs_mem"

    result = await openai_responses_impl.create_openai_response(
        input="repo prefs",
        model="test-model",
        stream=True,
        memory=MemoryToolConfig(enabled=False, owner_id="user-123"),
    )
    _chunks = [chunk async for chunk in result]

    mock_vector_io_api.openai_search_vector_store.assert_not_called()
