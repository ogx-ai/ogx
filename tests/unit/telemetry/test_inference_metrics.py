# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Unit tests for inference metrics."""

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ogx.core.routers.inference import InferenceRouter
from ogx.telemetry.inference_metrics import (
    create_inference_metric_attributes,
    inference_duration,
    inference_time_to_first_token,
    inference_tokens_per_second,
)
from ogx_api import (
    ModelType,
    OpenAIChatCompletion,
    OpenAIChatCompletionChunk,
    OpenAIChatCompletionRequestWithExtraBody,
    OpenAIChatCompletionResponseMessage,
    OpenAIChoice,
)


class TestInferenceMetricAttributes:
    """Test metric attribute creation utility."""

    def test_all_fields(self):
        attrs = create_inference_metric_attributes(
            model="openai/gpt-4o-mini",
            provider="openai",
            stream=True,
            status="success",
        )
        assert attrs == {
            "model": "openai/gpt-4o-mini",
            "provider": "openai",
            "stream": "true",
            "status": "success",
        }

    def test_partial_fields(self):
        attrs = create_inference_metric_attributes(
            model="openai/gpt-4o-mini",
            status="error",
        )
        assert attrs == {
            "model": "openai/gpt-4o-mini",
            "status": "error",
        }
        assert "provider" not in attrs
        assert "stream" not in attrs

    def test_empty(self):
        attrs = create_inference_metric_attributes()
        assert attrs == {}

    def test_stream_false(self):
        attrs = create_inference_metric_attributes(stream=False)
        assert attrs == {"stream": "false"}


class TestInferenceMetricInstruments:
    """Test that metric instruments are properly defined."""

    def test_inference_duration_exists(self):
        assert inference_duration is not None
        assert hasattr(inference_duration, "record")

    def test_inference_time_to_first_token_exists(self):
        assert inference_time_to_first_token is not None
        assert hasattr(inference_time_to_first_token, "record")

    def test_inference_tokens_per_second_exists(self):
        assert inference_tokens_per_second is not None
        assert hasattr(inference_tokens_per_second, "record")

    def test_inference_duration_can_record(self):
        attrs = create_inference_metric_attributes(
            model="openai/gpt-4o-mini",
            provider="openai",
            stream=False,
            status="success",
        )
        inference_duration.record(1.234, attrs)

    def test_inference_time_to_first_token_can_record(self):
        attrs = create_inference_metric_attributes(
            model="openai/gpt-4o-mini",
            provider="openai",
            stream=True,
            status="success",
        )
        inference_time_to_first_token.record(0.123, attrs)

    def test_inference_tokens_per_second_can_record(self):
        attrs = create_inference_metric_attributes(
            model="openai/gpt-4o-mini",
            provider="openai",
            stream=True,
            status="success",
        )
        inference_tokens_per_second.record(42.5, attrs)


class TestInferenceMetricsConstants:
    """Test that metric constants are properly defined."""

    def test_metric_names_follow_convention(self):
        from ogx.telemetry.constants import (
            INFERENCE_DURATION,
            INFERENCE_TIME_TO_FIRST_TOKEN,
            INFERENCE_TOKENS_PER_SECOND,
        )

        assert INFERENCE_DURATION.startswith("ogx.")
        assert INFERENCE_TIME_TO_FIRST_TOKEN.startswith("ogx.")
        assert INFERENCE_TOKENS_PER_SECOND.startswith("ogx.")

        assert "inference" in INFERENCE_DURATION
        assert "inference" in INFERENCE_TIME_TO_FIRST_TOKEN
        assert "inference" in INFERENCE_TOKENS_PER_SECOND

        assert INFERENCE_DURATION.endswith("_seconds")
        assert INFERENCE_TIME_TO_FIRST_TOKEN.endswith("_seconds")


def _make_router_and_provider():
    """Create a mock routing table and provider for testing."""
    routing_table = MagicMock()

    mock_model = MagicMock()
    mock_model.identifier = "openai/gpt-4o-mini"
    mock_model.model_type = ModelType.llm
    mock_model.provider_resource_id = "gpt-4o-mini"

    mock_provider = AsyncMock()
    mock_provider.__provider_id__ = "openai"

    routing_table.get_object_by_identifier = AsyncMock(return_value=mock_model)
    routing_table.get_provider_impl = AsyncMock(return_value=mock_provider)

    router = InferenceRouter(routing_table=routing_table)
    return router, mock_provider


def _make_chat_params(**kwargs):
    """Create minimal chat completion params."""
    defaults = {
        "model": "openai/gpt-4o-mini",
        "messages": [{"role": "user", "content": "Hello"}],
    }
    defaults.update(kwargs)
    return OpenAIChatCompletionRequestWithExtraBody(**defaults)


def _make_completion_response(**kwargs):
    """Create a minimal non-streaming chat completion response."""
    defaults = {
        "id": "chatcmpl-123",
        "choices": [
            OpenAIChoice(
                index=0,
                finish_reason="stop",
                message=OpenAIChatCompletionResponseMessage(
                    role="assistant",
                    content="Hello!",
                ),
            )
        ],
        "created": int(time.time()),
        "model": "gpt-4o-mini",
        "object": "chat.completion",
    }
    defaults.update(kwargs)
    return OpenAIChatCompletion(**defaults)


class TestNonStreamingInferenceMetrics:
    """Test that non-streaming chat completions record metrics."""

    async def test_records_duration_on_success(self):
        router, mock_provider = _make_router_and_provider()
        mock_provider.openai_chat_completion = AsyncMock(return_value=_make_completion_response())
        params = _make_chat_params(stream=False)

        with patch.object(inference_duration, "record") as mock_record:
            await router.openai_chat_completion(params)

            mock_record.assert_called_once()
            duration_val = mock_record.call_args[0][0]
            attrs = mock_record.call_args[1]["attributes"]
            assert duration_val > 0
            assert attrs["model"] == "openai/gpt-4o-mini"
            assert attrs["provider"] == "openai"
            assert attrs["stream"] == "false"
            assert attrs["status"] == "success"

    async def test_records_duration_on_error(self):
        router, mock_provider = _make_router_and_provider()
        mock_provider.openai_chat_completion = AsyncMock(side_effect=RuntimeError("provider error"))
        params = _make_chat_params(stream=False)

        with patch.object(inference_duration, "record") as mock_record:
            with pytest.raises(RuntimeError, match="provider error"):
                await router.openai_chat_completion(params)

            mock_record.assert_called_once()
            attrs = mock_record.call_args[1]["attributes"]
            assert attrs["status"] == "error"

    async def test_records_tokens_per_second_when_usage_present(self):
        from ogx_api.inference.models import OpenAIChatCompletionUsage

        router, mock_provider = _make_router_and_provider()
        usage = OpenAIChatCompletionUsage(completion_tokens=50, prompt_tokens=10, total_tokens=60)
        response = _make_completion_response(usage=usage)
        mock_provider.openai_chat_completion = AsyncMock(return_value=response)
        params = _make_chat_params(stream=False)

        with patch.object(inference_tokens_per_second, "record") as mock_record:
            await router.openai_chat_completion(params)

            mock_record.assert_called_once()
            tps_val = mock_record.call_args[0][0]
            assert tps_val > 0
            attrs = mock_record.call_args[1]["attributes"]
            assert attrs["status"] == "success"

    async def test_no_tokens_per_second_without_usage(self):
        router, mock_provider = _make_router_and_provider()
        response = _make_completion_response(usage=None)
        mock_provider.openai_chat_completion = AsyncMock(return_value=response)
        params = _make_chat_params(stream=False)

        with patch.object(inference_tokens_per_second, "record") as mock_record:
            await router.openai_chat_completion(params)
            mock_record.assert_not_called()


async def _make_streaming_chunks(chunks):
    """Create an async iterator from a list of chunks."""
    for chunk in chunks:
        yield chunk


def _make_chunk(
    chunk_id="chatcmpl-123",
    content=None,
    finish_reason=None,
    usage=None,
):
    """Create a minimal streaming chunk."""
    from ogx_api.inference.models import OpenAIChoiceDelta, OpenAIChunkChoice

    delta = OpenAIChoiceDelta(content=content, role="assistant" if content else None)
    choices = [OpenAIChunkChoice(index=0, delta=delta, finish_reason=finish_reason)]

    return OpenAIChatCompletionChunk(
        id=chunk_id,
        choices=choices,
        created=int(time.time()),
        model="gpt-4o-mini",
        object="chat.completion.chunk",
        usage=usage,
    )


class TestStreamingInferenceMetrics:
    """Test that streaming chat completions record metrics."""

    async def test_records_duration(self):
        router, mock_provider = _make_router_and_provider()
        chunks = [
            _make_chunk(content="Hello"),
            _make_chunk(content=" world"),
            _make_chunk(finish_reason="stop"),
        ]
        mock_provider.openai_chat_completion = AsyncMock(return_value=_make_streaming_chunks(chunks))
        params = _make_chat_params(stream=True)

        with patch.object(inference_duration, "record") as mock_record:
            stream = await router.openai_chat_completion(params)
            async for _ in stream:
                pass

            mock_record.assert_called_once()
            duration_val = mock_record.call_args[0][0]
            attrs = mock_record.call_args[1]["attributes"]
            assert duration_val > 0
            assert attrs["stream"] == "true"
            assert attrs["status"] == "success"

    async def test_records_ttft_on_first_content(self):
        router, mock_provider = _make_router_and_provider()
        chunks = [
            _make_chunk(content="Hello"),
            _make_chunk(content=" world"),
            _make_chunk(finish_reason="stop"),
        ]
        mock_provider.openai_chat_completion = AsyncMock(return_value=_make_streaming_chunks(chunks))
        params = _make_chat_params(stream=True)

        with patch.object(inference_time_to_first_token, "record") as mock_record:
            stream = await router.openai_chat_completion(params)
            async for _ in stream:
                pass

            mock_record.assert_called_once()
            ttft_val = mock_record.call_args[0][0]
            assert ttft_val >= 0
            attrs = mock_record.call_args[1]["attributes"]
            assert attrs["stream"] == "true"

    async def test_no_ttft_without_content(self):
        router, mock_provider = _make_router_and_provider()
        chunks = [
            _make_chunk(finish_reason="stop"),
        ]
        mock_provider.openai_chat_completion = AsyncMock(return_value=_make_streaming_chunks(chunks))
        params = _make_chat_params(stream=True)

        with patch.object(inference_time_to_first_token, "record") as mock_record:
            stream = await router.openai_chat_completion(params)
            async for _ in stream:
                pass

            mock_record.assert_not_called()

    async def test_records_tokens_per_second_from_usage(self):
        router, mock_provider = _make_router_and_provider()
        from ogx_api.inference.models import OpenAIChatCompletionUsage

        usage = OpenAIChatCompletionUsage(completion_tokens=100, prompt_tokens=10, total_tokens=110)
        chunks = [
            _make_chunk(content="Hello"),
            _make_chunk(finish_reason="stop", usage=usage),
        ]
        mock_provider.openai_chat_completion = AsyncMock(return_value=_make_streaming_chunks(chunks))
        params = _make_chat_params(stream=True)

        with patch.object(inference_tokens_per_second, "record") as mock_record:
            stream = await router.openai_chat_completion(params)
            async for _ in stream:
                pass

            mock_record.assert_called_once()
            tps_val = mock_record.call_args[0][0]
            assert tps_val > 0

    async def test_no_tokens_per_second_without_usage(self):
        router, mock_provider = _make_router_and_provider()
        chunks = [
            _make_chunk(content="Hello"),
            _make_chunk(finish_reason="stop"),
        ]
        mock_provider.openai_chat_completion = AsyncMock(return_value=_make_streaming_chunks(chunks))
        params = _make_chat_params(stream=True)

        with patch.object(inference_tokens_per_second, "record") as mock_record:
            stream = await router.openai_chat_completion(params)
            async for _ in stream:
                pass

            mock_record.assert_not_called()

    async def test_records_error_status_on_exception(self):
        router, mock_provider = _make_router_and_provider()

        async def failing_stream():
            yield _make_chunk(content="Hello")
            raise RuntimeError("stream error")

        mock_provider.openai_chat_completion = AsyncMock(return_value=failing_stream())
        params = _make_chat_params(stream=True)

        with patch.object(inference_duration, "record") as mock_record:
            stream = await router.openai_chat_completion(params)
            with pytest.raises(RuntimeError, match="stream error"):
                async for _ in stream:
                    pass

            mock_record.assert_called_once()
            attrs = mock_record.call_args[1]["attributes"]
            assert attrs["status"] == "error"
