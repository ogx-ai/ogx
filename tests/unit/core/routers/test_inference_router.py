# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""
Unit tests for InferenceRouter to verify correct provider method invocation.

Test Categories:
1. Rerank method routing - validates that rerank calls are properly routed to providers
2. Model resolution - validates model to provider mapping
3. Parameter transformation - validates request object modifications for provider calls

Specific Tests:
- test_rerank_calls_provider_correctly: Validates the router calls provider.rerank() with correct RerankRequest
"""

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from openai.types.chat import ChatCompletionChunk
from pydantic import ValidationError

from ogx.core.request_headers import PROVIDER_DATA_VAR
from ogx.core.routers.inference import InferenceRouter, _normalize_provider_data_key
from ogx.core.storage.datatypes import InferenceStoreReference, SqliteSqlStoreConfig
from ogx.core.storage.sqlstore.sqlstore import register_sqlstore_backends
from ogx.providers.utils.inference.inference_store import InferenceStore
from ogx.providers.utils.inference.model_registry import RemoteInferenceProviderConfig
from ogx_api import (
    GetChatCompletionRequest,
    ListChatCompletionMessagesRequest,
    ListChatCompletionsRequest,
    ModelType,
    ModelTypeError,
    OpenAICompletion,
    OpenAICompletionRequestWithExtraBody,
    RerankData,
    RerankResponse,
    RoutingTable,
    SystemOneRequest,
    SystemOneResponse,
)
from ogx_api.inference import RerankRequest
from ogx_api.inference.models import (
    OpenAIChatCompletion,
    OpenAIChatCompletionChunk,
    OpenAIChatCompletionRequestWithExtraBody,
    OpenAIChatCompletionResponseMessage,
    OpenAIChoice,
    OpenAIChoiceDelta,
    OpenAIChunkChoice,
    OpenAICompletionChoice,
)
from tests.unit.providers.utils.inference.openai_mixin_helpers import OpenAIMixinWithProviderData


@pytest.fixture
def mock_routing_table():
    """Create a mock routing table with model and provider setup"""
    routing_table = MagicMock(spec=RoutingTable)

    mock_model = MagicMock()
    mock_model.identifier = "test-rerank-model"
    mock_model.model_type = ModelType.rerank
    mock_model.provider_resource_id = "provider-rerank-model-123"

    mock_provider = MagicMock()
    mock_provider.__provider_id__ = "test_provider"

    routing_table.get_object_by_identifier = AsyncMock(return_value=mock_model)
    routing_table.get_provider_impl = AsyncMock(return_value=mock_provider)

    return routing_table, mock_provider


@pytest.fixture
def mock_llm_routing_table():
    """Create a mock routing table with an LLM model registered under a fully qualified id"""
    routing_table = MagicMock(spec=RoutingTable)

    mock_model = MagicMock()
    mock_model.identifier = "test_provider/test-llm-model"
    mock_model.model_type = ModelType.llm
    mock_model.provider_resource_id = "test-llm-model"

    mock_provider = MagicMock()
    mock_provider.__provider_id__ = "test_provider"

    routing_table.get_object_by_identifier = AsyncMock(return_value=mock_model)
    routing_table.get_provider_impl = AsyncMock(return_value=mock_provider)

    return routing_table, mock_provider


@pytest.fixture
def mock_systemone_routing_table():
    """Create a mock routing table with a SystemOne (judgment) model and provider."""
    routing_table = MagicMock(spec=RoutingTable)

    mock_model = MagicMock()
    mock_model.identifier = "typesafe/jev-latest"
    mock_model.model_type = ModelType.systemone
    mock_model.provider_resource_id = "jev-latest"

    mock_provider = MagicMock()
    mock_provider.__provider_id__ = "typesafe"

    routing_table.get_object_by_identifier = AsyncMock(return_value=mock_model)
    routing_table.get_provider_impl = AsyncMock(return_value=mock_provider)

    return routing_table, mock_provider


def _make_completion_chunk(text: str, model: str) -> OpenAICompletion:
    return OpenAICompletion(
        id="cmpl-test",
        choices=[OpenAICompletionChoice(finish_reason="stop", text=text, index=0)],
        created=0,
        model=model,
        object="text_completion",
    )


async def test_openai_completion_streaming_rewrites_model_id(mock_llm_routing_table):
    """
    Test that streamed /v1/completions chunks report the fully qualified model id
    that the client requested, not the provider-internal resource id.

    This mirrors the non-streaming path in openai_completion (which sets
    response.model = request_model_id) and the chat streaming path
    (stream_tokens_and_compute_metrics_openai_chat, which rewrites chunk.model).
    """
    routing_table, mock_provider = mock_llm_routing_table
    router = InferenceRouter(routing_table=routing_table)

    async def provider_stream():
        # Providers respond with their internal model id
        yield _make_completion_chunk("Hello", model="test-llm-model")
        yield _make_completion_chunk(" world", model="test-llm-model")

    mock_provider.openai_completion = AsyncMock(return_value=provider_stream())

    params = OpenAICompletionRequestWithExtraBody(
        model="test_provider/test-llm-model",
        prompt="Say hello",
        stream=True,
    )

    stream = await router.openai_completion(params)
    chunks = [chunk async for chunk in stream]

    assert len(chunks) == 2
    assert [chunk.model for chunk in chunks] == ["test_provider/test-llm-model", "test_provider/test-llm-model"], (
        "Streamed completion chunks should carry the requested model id, not the provider resource id"
    )
    assert [choice.text for chunk in chunks for choice in chunk.choices] == ["Hello", " world"]

    # The provider itself should still be called with its own resource id
    called_params = mock_provider.openai_completion.call_args.args[0]
    assert called_params.model == "test-llm-model"


async def test_openai_completion_streaming_empty_stream(mock_llm_routing_table):
    """A provider stream that yields no chunks produces an empty stream without errors."""
    routing_table, mock_provider = mock_llm_routing_table
    router = InferenceRouter(routing_table=routing_table)

    async def provider_stream():
        return
        yield  # unreachable; makes this function an async generator

    mock_provider.openai_completion = AsyncMock(return_value=provider_stream())

    params = OpenAICompletionRequestWithExtraBody(
        model="test_provider/test-llm-model",
        prompt="Say hello",
        stream=True,
    )

    stream = await router.openai_completion(params)
    chunks = [chunk async for chunk in stream]

    assert chunks == []


async def test_openai_completion_streaming_model_id_already_correct(mock_llm_routing_table):
    """Chunks that already carry the fully qualified model id are passed through unchanged."""
    routing_table, mock_provider = mock_llm_routing_table
    router = InferenceRouter(routing_table=routing_table)

    async def provider_stream():
        yield _make_completion_chunk("Hello", model="test_provider/test-llm-model")

    mock_provider.openai_completion = AsyncMock(return_value=provider_stream())

    params = OpenAICompletionRequestWithExtraBody(
        model="test_provider/test-llm-model",
        prompt="Say hello",
        stream=True,
    )

    stream = await router.openai_completion(params)
    chunks = [chunk async for chunk in stream]

    assert len(chunks) == 1
    assert chunks[0].model == "test_provider/test-llm-model"
    assert chunks[0].choices[0].text == "Hello"


async def test_openai_completion_streaming_skips_none_chunks(mock_llm_routing_table):
    """None chunks from a provider are skipped, mirroring the chat streaming path."""
    routing_table, mock_provider = mock_llm_routing_table
    router = InferenceRouter(routing_table=routing_table)

    async def provider_stream():
        yield _make_completion_chunk("Hello", model="test-llm-model")
        yield None
        yield _make_completion_chunk(" world", model="test-llm-model")

    mock_provider.openai_completion = AsyncMock(return_value=provider_stream())

    params = OpenAICompletionRequestWithExtraBody(
        model="test_provider/test-llm-model",
        prompt="Say hello",
        stream=True,
    )

    stream = await router.openai_completion(params)
    chunks = [chunk async for chunk in stream]

    assert [chunk.model for chunk in chunks] == ["test_provider/test-llm-model", "test_provider/test-llm-model"]
    assert [choice.text for chunk in chunks for choice in chunk.choices] == ["Hello", " world"]


async def test_openai_completion_streaming_propagates_provider_errors(mock_llm_routing_table):
    """Errors raised by the provider mid-stream propagate to the caller after earlier chunks are delivered."""
    routing_table, mock_provider = mock_llm_routing_table
    router = InferenceRouter(routing_table=routing_table)

    async def provider_stream():
        yield _make_completion_chunk("Hello", model="test-llm-model")
        raise RuntimeError("provider stream failed")

    mock_provider.openai_completion = AsyncMock(return_value=provider_stream())

    params = OpenAICompletionRequestWithExtraBody(
        model="test_provider/test-llm-model",
        prompt="Say hello",
        stream=True,
    )

    stream = await router.openai_completion(params)
    chunks = []
    with pytest.raises(RuntimeError, match="provider stream failed"):
        async for chunk in stream:
            chunks.append(chunk)

    assert len(chunks) == 1
    assert chunks[0].model == "test_provider/test-llm-model"


async def test_openai_completion_non_streaming_rewrites_model_id(mock_llm_routing_table):
    """Non-streaming /v1/completions responses report the requested model id (regression guard)."""
    routing_table, mock_provider = mock_llm_routing_table
    router = InferenceRouter(routing_table=routing_table)

    mock_provider.openai_completion = AsyncMock(
        return_value=_make_completion_chunk("Hello world", model="test-llm-model")
    )

    params = OpenAICompletionRequestWithExtraBody(
        model="test_provider/test-llm-model",
        prompt="Say hello",
    )

    response = await router.openai_completion(params)

    assert response.model == "test_provider/test-llm-model"


async def test_rerank_calls_provider_correctly(mock_routing_table):
    """
    Test that InferenceRouter.rerank() calls the provider's rerank method with the correct RerankRequest.

    This test validates:
    - The provider's rerank method is called exactly once
    - The provider receives a RerankRequest object (not individual parameters)
    - The model ID is substituted with provider_resource_id
    """
    routing_table, mock_provider = mock_routing_table
    router = InferenceRouter(routing_table=routing_table)

    expected_response = RerankResponse(
        data=[
            RerankData(index=0, relevance_score=0.9),
        ]
    )
    mock_provider.rerank = AsyncMock(return_value=expected_response)

    request = RerankRequest(
        model="test-rerank-model",
        query="test query",
        items=["item1", "item2"],
        max_num_results=1,
    )

    result = await router.rerank(request)

    mock_provider.rerank.assert_called_once()

    call_args = mock_provider.rerank.call_args
    assert len(call_args.args) == 1, "Provider.rerank should be called with exactly one argument"
    assert isinstance(call_args.args[0], RerankRequest), "Provider.rerank should receive a RerankRequest object"

    called_request = call_args.args[0]
    assert called_request.model == "provider-rerank-model-123", "Model should be substituted with provider_resource_id"

    assert called_request.query == "test query"
    assert called_request.items == ["item1", "item2"]
    assert called_request.max_num_results == 1

    assert result == expected_response


async def test_judge_calls_provider_correctly(mock_systemone_routing_table):
    """InferenceRouter.judge() routes a SystemOne judgment to the model's provider.

    Validates the model is resolved as ModelType.systemone, the request model id is rewritten to
    the provider resource id for the provider call, and the response model is rewritten back to
    the fully-qualified id the client requested (so the provider's internal id isn't leaked).
    """
    routing_table, mock_provider = mock_systemone_routing_table
    router = InferenceRouter(routing_table=routing_table)

    expected_response = SystemOneResponse(
        model="jev-latest",
        answers={"billing": {"type": "noul", "noul": 0.9}},
        usage={"input_tokens": 12, "output_tokens": 4},
    )
    mock_provider.judge = AsyncMock(return_value=expected_response)

    request = SystemOneRequest(
        model="typesafe/jev-latest",
        state="I was charged twice. Please help.",
        questions={"billing": {"type": "noul", "instructions": "Is this about billing?"}},
    )

    result = await router.judge(request)

    mock_provider.judge.assert_called_once()
    called_request = mock_provider.judge.call_args.args[0]
    assert isinstance(called_request, SystemOneRequest)
    assert called_request.model == "jev-latest", "Model should be rewritten to the provider resource id"
    # The provider returns its internal id ("jev-latest"); the router must rewrite the response
    # model back to the fully-qualified id the client requested so the provider id isn't leaked.
    assert result.model == "typesafe/jev-latest", "Response model must be the requested id, not the provider id"
    assert result.answers == expected_response.answers
    assert result.usage == expected_response.usage


async def test_judge_rejects_non_systemone_model(mock_systemone_routing_table):
    """A model that is not ModelType.systemone is rejected before the provider is called."""
    routing_table, mock_provider = mock_systemone_routing_table
    routing_table.get_object_by_identifier = AsyncMock(
        return_value=MagicMock(
            identifier="typesafe/gpt-4o",
            model_type=ModelType.llm,
            provider_resource_id="gpt-4o",
        )
    )
    router = InferenceRouter(routing_table=routing_table)

    request = SystemOneRequest(
        model="typesafe/gpt-4o",
        state="state",
        questions={"billing": {"type": "noul", "instructions": "Is this billing?"}},
    )

    with pytest.raises(ModelTypeError):
        await router.judge(request)
    mock_provider.judge.assert_not_called()


def test_systemone_choice_criteria_is_bounded_to_255():
    """A `choice` question accepts up to 255 options but rejects 256."""

    def _choice_request(n_options: int) -> SystemOneRequest:
        return SystemOneRequest(
            model="m",
            state="s",
            questions={
                "q": {"type": "choice", "instructions": "pick", "criteria": {str(i): None for i in range(n_options)}}
            },
        )

    assert _choice_request(255).questions["q"].type == "choice"
    with pytest.raises(ValidationError):
        _choice_request(256)


def _make_chat_completion(
    model: str = "test_provider/test-llm-model",
) -> OpenAIChatCompletion:
    """Build a minimal non-streaming chat completion response."""
    return OpenAIChatCompletion(
        id="chatcmpl-test",
        choices=[
            OpenAIChoice(
                message=OpenAIChatCompletionResponseMessage(role="assistant", content="Hello world"),
                finish_reason="stop",
                index=0,
            )
        ],
        created=0,
        model=model,
    )


def _make_chat_completion_chunk(text: str, model: str = "test-llm-model") -> OpenAIChatCompletionChunk:
    """Build a minimal streaming chat completion chunk."""
    return OpenAIChatCompletionChunk(
        id="chatcmpl-test",
        choices=[
            OpenAIChunkChoice(
                delta=OpenAIChoiceDelta(content=text, role="assistant"),
                finish_reason=None,
                index=0,
            )
        ],
        created=0,
        model=model,
    )


async def test_openai_chat_completion_non_streaming_without_store(
    mock_llm_routing_table: tuple[MagicMock, MagicMock],
) -> None:
    """A non-streaming chat completion succeeds when persistence is disabled (no store).

    The completion is returned to the caller with the requested model id and the
    router never attempts to store it (there is no store).
    """
    routing_table, mock_provider = mock_llm_routing_table
    router = InferenceRouter(routing_table=routing_table)
    assert router.store is None

    mock_provider.openai_chat_completion = AsyncMock(return_value=_make_chat_completion())

    params = OpenAIChatCompletionRequestWithExtraBody(
        model="test_provider/test-llm-model",
        messages=[{"role": "user", "content": "Say hello"}],
    )

    response = await router.openai_chat_completion(params)

    assert response.id == "chatcmpl-test"
    assert response.model == "test_provider/test-llm-model"
    assert response.choices[0].message.content == "Hello world"
    # Provider was called with its resource id, not the fully qualified id.
    assert mock_provider.openai_chat_completion.call_args.args[0].model == "test-llm-model"


async def test_openai_chat_completion_streaming_without_store(
    mock_llm_routing_table: tuple[MagicMock, MagicMock],
) -> None:
    """A streaming chat completion streams normally when persistence is disabled.

    Chunks are rewritten to carry the requested model id and the router never
    attempts to assemble/store a final completion (there is no store).
    """
    routing_table, mock_provider = mock_llm_routing_table
    router = InferenceRouter(routing_table=routing_table)
    assert router.store is None

    async def provider_stream() -> AsyncIterator[OpenAIChatCompletionChunk]:
        yield _make_chat_completion_chunk("Hello")
        yield _make_chat_completion_chunk(" world")

    mock_provider.openai_chat_completion = AsyncMock(return_value=provider_stream())

    params = OpenAIChatCompletionRequestWithExtraBody(
        model="test_provider/test-llm-model",
        messages=[{"role": "user", "content": "Say hello"}],
        stream=True,
    )

    stream = await router.openai_chat_completion(params)
    chunks = [chunk async for chunk in stream]

    assert len(chunks) == 2
    assert [chunk.model for chunk in chunks] == [
        "test_provider/test-llm-model",
        "test_provider/test-llm-model",
    ]
    assert ["".join(c.delta.content or "" for c in chunk.choices) for chunk in chunks] == ["Hello", " world"]


_URL_CITATION = {
    "type": "url_citation",
    "url_citation": {"start_index": 0, "end_index": 5, "title": "Example", "url": "https://example.com"},
}


def test_chunk_delta_accepts_annotations() -> None:
    """Search models stream url_citation annotations in the chunk delta."""
    chunk = OpenAIChatCompletionChunk.model_validate(
        {
            "id": "chatcmpl-test",
            "created": 0,
            "model": "test-llm-model",
            "choices": [{"index": 0, "delta": {"annotations": [_URL_CITATION]}}],
        }
    )

    assert chunk.choices[0].delta.annotations == [_URL_CITATION]
    assert chunk.model_dump()["choices"][0]["delta"]["annotations"] == [_URL_CITATION]


async def test_streamed_chat_completion_is_stored_with_annotations(
    mock_llm_routing_table: tuple[MagicMock, MagicMock], tmp_path: Path
) -> None:
    """The completion stored after a stream keeps the annotations streamed in the deltas."""
    register_sqlstore_backends({"sql_default": SqliteSqlStoreConfig(db_path=str(tmp_path / "store.db"))})
    store = InferenceStore(InferenceStoreReference(backend="sql_default", table_name="chat_completions"), policy=[])
    await store.initialize()
    routing_table, mock_provider = mock_llm_routing_table
    router = InferenceRouter(routing_table=routing_table, store=store)

    def sdk_chunk(delta: dict, finish_reason: str | None = None) -> ChatCompletionChunk:
        # OpenAI-compatible providers yield the SDK's chunk type, not ours
        return ChatCompletionChunk.model_validate(
            {
                "id": "chatcmpl-test",
                "object": "chat.completion.chunk",
                "created": 0,
                "model": "test-llm-model",
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
            }
        )

    async def provider_stream() -> AsyncIterator[ChatCompletionChunk]:
        yield sdk_chunk({"role": "assistant", "content": "Hello"})
        yield sdk_chunk({"annotations": [_URL_CITATION]})
        yield sdk_chunk({}, "stop")

    mock_provider.openai_chat_completion = AsyncMock(return_value=provider_stream())

    params = OpenAIChatCompletionRequestWithExtraBody(
        model="test_provider/test-llm-model",
        messages=[{"role": "user", "content": "Say hello"}],
        stream=True,
    )

    stream = await router.openai_chat_completion(params)
    _ = [chunk async for chunk in stream]
    # The router stores the completion in a background task
    await asyncio.gather(*(asyncio.all_tasks() - {asyncio.current_task()}))

    stored = await router.get_chat_completion(GetChatCompletionRequest(completion_id="chatcmpl-test"))
    assert stored.choices[0].message.content == "Hello"
    assert stored.choices[0].message.annotations == [_URL_CITATION]


async def test_list_chat_completions_without_store_raises_not_implemented(
    mock_llm_routing_table: tuple[MagicMock, MagicMock],
) -> None:
    """The list history endpoint reports an error (not an empty list) when persistence is off."""
    routing_table, _ = mock_llm_routing_table
    router = InferenceRouter(routing_table=routing_table)
    assert router.store is None

    with pytest.raises(NotImplementedError):
        await router.list_chat_completions(ListChatCompletionsRequest())


async def test_get_chat_completion_without_store_raises_not_implemented(
    mock_llm_routing_table: tuple[MagicMock, MagicMock],
) -> None:
    """The retrieve history endpoint reports an error (not a 404) when persistence is off."""
    routing_table, _ = mock_llm_routing_table
    router = InferenceRouter(routing_table=routing_table)
    assert router.store is None

    with pytest.raises(NotImplementedError):
        await router.get_chat_completion(GetChatCompletionRequest(completion_id="chatcmpl-test"))


async def test_list_chat_completion_messages_without_store_raises_not_implemented(
    mock_llm_routing_table: tuple[MagicMock, MagicMock],
) -> None:
    """The messages history endpoint reports an error when persistence is off, consistent with list/retrieve."""
    routing_table, _ = mock_llm_routing_table
    router = InferenceRouter(routing_table=routing_table)
    assert router.store is None

    with pytest.raises(NotImplementedError):
        await router.list_chat_completion_messages(ListChatCompletionMessagesRequest(completion_id="chatcmpl-test"))


# ---------------------------------------------------------------------------
# Provider-data key normalization: {provider_id}_api_key -> impl key
# ---------------------------------------------------------------------------


def _fake_provider(provider_id: str | None, impl_key: str | None) -> MagicMock:
    provider = MagicMock()
    provider.__provider_id__ = provider_id
    provider.provider_data_api_key_field = impl_key
    return provider


@pytest.mark.parametrize(
    ("provider_id", "impl_key", "initial", "expected"),
    [
        # Client key is mapped to the impl key; the client key is retained.
        (
            "providerA",
            "vllm_api_token",
            {"providerA_api_key": "tok"},
            {"providerA_api_key": "tok", "vllm_api_token": "tok"},
        ),
        # No provider data on the request -> contextvar left untouched.
        ("providerA", "vllm_api_token", None, None),
        # Client key absent -> nothing to map.
        ("providerA", "vllm_api_token", {"other": 1}, {"other": 1}),
        # No provider id -> cannot derive the client key.
        (None, "vllm_api_token", {"providerA_api_key": "tok"}, {"providerA_api_key": "tok"}),
        # Provider has no impl key (not a provider-data provider) -> no-op.
        ("providerA", None, {"providerA_api_key": "tok"}, {"providerA_api_key": "tok"}),
        # Client key takes precedence over a pre-existing legacy key.
        (
            "providerA",
            "vllm_api_token",
            {"providerA_api_key": "new", "vllm_api_token": "old"},
            {"providerA_api_key": "new", "vllm_api_token": "new"},
        ),
    ],
)
def test_normalize_provider_data_key(provider_id, impl_key, initial, expected):
    provider = _fake_provider(provider_id, impl_key)
    token = PROVIDER_DATA_VAR.set(dict(initial) if initial is not None else None)
    try:
        _normalize_provider_data_key(provider)
        assert PROVIDER_DATA_VAR.get() == expected
    finally:
        PROVIDER_DATA_VAR.reset(token)


def test_normalize_provider_data_key_preserves_unrelated_keys():
    provider = _fake_provider("providerA", "vllm_api_token")
    initial = {"providerA_api_key": "tok", "__authenticated_user": {"id": "u1"}, "unrelated": "x"}
    token = PROVIDER_DATA_VAR.set(dict(initial))
    try:
        _normalize_provider_data_key(provider)
        result = PROVIDER_DATA_VAR.get()
        assert result["vllm_api_token"] == "tok"
        assert result["__authenticated_user"] == {"id": "u1"}
        assert result["unrelated"] == "x"
    finally:
        PROVIDER_DATA_VAR.reset(token)


@pytest.mark.parametrize(
    "client_key", ["providerA_api_key", "PROVIDERA_API_KEY", "Providera_Api_Key", "providera_api_key"]
)
def test_normalize_provider_data_key_is_case_insensitive(client_key):
    provider = _fake_provider("providerA", "vllm_api_token")
    token = PROVIDER_DATA_VAR.set({client_key: "tok"})
    try:
        _normalize_provider_data_key(provider)
        result = PROVIDER_DATA_VAR.get()
        assert result[client_key] == "tok"
        assert result["vllm_api_token"] == "tok"
    finally:
        PROVIDER_DATA_VAR.reset(token)


def test_normalize_provider_data_key_ignores_legacy_key_casing():
    """A differently-cased legacy impl key is not treated as the client key."""
    provider = _fake_provider("providerA", "vllm_api_token")
    token = PROVIDER_DATA_VAR.set({"VLLM_API_TOKEN": "legacy"})
    try:
        _normalize_provider_data_key(provider)
        assert PROVIDER_DATA_VAR.get() == {"VLLM_API_TOKEN": "legacy"}
    finally:
        PROVIDER_DATA_VAR.reset(token)


@pytest.mark.parametrize(
    "provider_data_key",
    [
        "providerA_api_key",  # client key, exact case
        "PROVIDERA_API_KEY",  # client key, upper case
        "Providera_Api_Key",  # client key, mixed case
        "test_api_key",  # fallback: provider_data_api_key_field, passes through unmapped
    ],
)
def test_client_api_key_reaches_openai_client_after_normalization(provider_data_key):
    """A per-request credential reaches the provider's OpenAI client api_key.

    The client key {provider_id}_api_key is matched case-insensitively and mapped to
    the impl key by the router; the legacy provider_data_api_key_field is a fallback
    that passes through unmapped. Uses a real (mock) OpenAI mixin so the client is
    built from the (normalized) provider-data key. No network is touched.
    """
    provider = OpenAIMixinWithProviderData(config=RemoteInferenceProviderConfig())
    provider.__provider_id__ = "providerA"
    provider.__provider_spec__ = MagicMock(
        provider_type="test",
        provider_data_validator="tests.unit.providers.utils.inference.openai_mixin_helpers.ProviderDataValidator",
    )

    token = PROVIDER_DATA_VAR.set({provider_data_key: "tok"})
    try:
        _normalize_provider_data_key(provider)
        assert provider.client.api_key == "tok"
    finally:
        PROVIDER_DATA_VAR.reset(token)


def test_client_key_overrides_provider_data_field():
    """When both the client key and the impl key are present, the client key wins.

    A client that sends {provider_id}_api_key takes precedence over a legacy
    provider_data_api_key_field value, so the client key's credential is the one
    that reaches the OpenAI client.
    """
    provider = OpenAIMixinWithProviderData(config=RemoteInferenceProviderConfig())
    provider.__provider_id__ = "providerA"
    provider.__provider_spec__ = MagicMock(
        provider_type="test",
        provider_data_validator="tests.unit.providers.utils.inference.openai_mixin_helpers.ProviderDataValidator",
    )

    token = PROVIDER_DATA_VAR.set({"providerA_api_key": "client_tok", "test_api_key": "impl_tok"})
    try:
        _normalize_provider_data_key(provider)
        assert provider.client.api_key == "client_tok"
    finally:
        PROVIDER_DATA_VAR.reset(token)


async def test_get_model_provider_normalizes_provider_data_key(mock_llm_routing_table):
    """The router normalizes {provider_id}_api_key before the provider is invoked."""
    routing_table, mock_provider = mock_llm_routing_table
    mock_provider.__provider_id__ = "test_provider"
    mock_provider.provider_data_api_key_field = "vllm_api_token"
    router = InferenceRouter(routing_table=routing_table)

    captured = {}

    async def fake_completion(params):
        captured["provider_data"] = dict(PROVIDER_DATA_VAR.get() or {})
        return _make_completion_chunk("hi", model="test-llm-model")

    mock_provider.openai_completion = AsyncMock(side_effect=fake_completion)

    token = PROVIDER_DATA_VAR.set({"test_provider_api_key": "tok"})
    try:
        params = OpenAICompletionRequestWithExtraBody(model="test_provider/test-llm-model", prompt="hi")
        await router.openai_completion(params)
    finally:
        PROVIDER_DATA_VAR.reset(token)

    assert captured["provider_data"].get("vllm_api_token") == "tok"
    assert captured["provider_data"].get("test_provider_api_key") == "tok"


async def test_get_model_provider_without_provider_data_is_unchanged(mock_llm_routing_table):
    """No provider data on the request leaves the provider data context empty."""
    routing_table, mock_provider = mock_llm_routing_table
    mock_provider.__provider_id__ = "test_provider"
    mock_provider.provider_data_api_key_field = "vllm_api_token"
    router = InferenceRouter(routing_table=routing_table)

    captured = {}

    async def fake_completion(params):
        captured["provider_data"] = dict(PROVIDER_DATA_VAR.get() or {})
        return _make_completion_chunk("hi", model="test-llm-model")

    mock_provider.openai_completion = AsyncMock(side_effect=fake_completion)

    token = PROVIDER_DATA_VAR.set(None)
    try:
        params = OpenAICompletionRequestWithExtraBody(model="test_provider/test-llm-model", prompt="hi")
        await router.openai_completion(params)
    finally:
        PROVIDER_DATA_VAR.reset(token)

    assert captured["provider_data"] == {}
