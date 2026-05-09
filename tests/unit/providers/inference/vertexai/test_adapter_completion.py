# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import json
import logging  # allow-direct-logging
from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest

from ogx.providers.remote.inference.vertexai.config import VertexAIConfig
from ogx.providers.remote.inference.vertexai.vertexai import VertexAIInferenceAdapter
from ogx_api import OpenAICompletion
from ogx_api.inference.models import OpenAICompletionRequestWithExtraBody


async def _async_pager(items):
    """Yield items through an async iterator."""
    for item in items:
        yield item


def _make_fake_gemini_response(text: str = "The answer") -> SimpleNamespace:
    """Build a fake Gemini response object."""
    part = SimpleNamespace(text=text, function_call=None, thought=None)
    content = SimpleNamespace(parts=[part])
    candidate = SimpleNamespace(content=content, finish_reason=None, index=0, logprobs_result=None)
    return SimpleNamespace(candidates=[candidate], usage_metadata=None)


def _make_fake_streaming_chunk(text: str = "chunk") -> SimpleNamespace:
    """Build a fake streaming chunk object."""
    part = SimpleNamespace(text=text, function_call=None, thought=None)
    content = SimpleNamespace(parts=[part])
    candidate = SimpleNamespace(content=content, finish_reason="STOP", index=0, logprobs_result=None)
    return SimpleNamespace(candidates=[candidate], usage_metadata=None)


@pytest.fixture
def make_completion_adapter(monkeypatch):
    """Create completion adapter."""

    def factory(fake_response=None):
        """Create an adapter with a mocked client."""
        if fake_response is None:
            fake_response = _make_fake_gemini_response()
        a = VertexAIInferenceAdapter(config=VertexAIConfig(project="p", location="l"))
        fake_client = MagicMock()
        fake_client.aio.models.generate_content = AsyncMock(return_value=fake_response)
        monkeypatch.setattr(a, "_get_provider_model_id", AsyncMock(return_value="gemini-2.5-flash"))
        monkeypatch.setattr(a, "_validate_model_allowed", lambda _: None)
        monkeypatch.setattr(a, "_get_client", lambda: fake_client)
        return a, fake_client

    return factory


class TestOpenAICompletion:
    async def test_string_prompt_returns_completion(self, make_completion_adapter):
        """Test that string prompt returns completion."""
        adapter, _ = make_completion_adapter()
        params = OpenAICompletionRequestWithExtraBody(model="google/gemini-2.5-flash", prompt="Hello world")
        result = await adapter.openai_completion(params)
        assert isinstance(result, OpenAICompletion)
        assert result.choices[0].text == "The answer"

    async def test_list_string_prompt_accepted(self, make_completion_adapter):
        """Test that list string prompt accepted."""
        adapter, _ = make_completion_adapter()
        params = OpenAICompletionRequestWithExtraBody(model="google/gemini-2.5-flash", prompt=["prompt1", "prompt2"])
        result = await adapter.openai_completion(params)
        assert isinstance(result, OpenAICompletion)
        assert len(result.choices) == 2

    async def test_token_array_prompt_raises_value_error(self, make_completion_adapter):
        """Test that token array prompt raises value error."""
        adapter, _ = make_completion_adapter()
        params = OpenAICompletionRequestWithExtraBody(model="google/gemini-2.5-flash", prompt=[1, 2, 3])
        with pytest.raises(ValueError, match="Token array"):
            await adapter.openai_completion(params)

    async def test_nested_token_array_prompt_raises_value_error(self, make_completion_adapter):
        """Test that nested token array prompt raises value error."""
        adapter, _ = make_completion_adapter()
        params = OpenAICompletionRequestWithExtraBody(model="google/gemini-2.5-flash", prompt=[[1, 2], [3, 4]])
        with pytest.raises(ValueError, match="Token array"):
            await adapter.openai_completion(params)

    async def test_stream_list_prompt_raises_value_error(self, monkeypatch):
        """Test that stream list prompt raises value error."""
        adapter = VertexAIInferenceAdapter(config=VertexAIConfig(project="p", location="l"))
        fake_chunk = _make_fake_streaming_chunk("hi")
        fake_client = MagicMock()
        fake_client.aio.models.generate_content_stream = AsyncMock(return_value=_async_pager([fake_chunk]))
        monkeypatch.setattr(adapter, "_get_provider_model_id", AsyncMock(return_value="gemini-2.5-flash"))
        monkeypatch.setattr(adapter, "_validate_model_allowed", lambda _: None)
        monkeypatch.setattr(adapter, "_get_client", lambda: fake_client)
        params = OpenAICompletionRequestWithExtraBody(model="google/gemini-2.5-flash", prompt=["a", "b"], stream=True)
        result = await adapter.openai_completion(params)
        assert hasattr(result, "__aiter__")

    async def test_unmappable_params_logged(self, make_completion_adapter, caplog):
        """Test that unmappable params logged."""
        adapter, _ = make_completion_adapter()
        params = OpenAICompletionRequestWithExtraBody(
            model="google/gemini-2.5-flash",
            prompt="hi",
            best_of=3,
            echo=True,
            suffix="end",
            logit_bias={"50256": -100},
        )
        with caplog.at_level(logging.WARNING, logger="ogx.providers.remote.inference.vertexai.vertexai"):
            await adapter.openai_completion(params)
        messages = " ".join(r.message for r in caplog.records)
        assert "best_of" in messages
        assert "suffix" in messages
        assert "logit_bias" in messages
        assert "echo" not in messages

    async def test_user_param_debug_log(self, make_completion_adapter, caplog):
        """Test that user param debug log."""
        adapter, _ = make_completion_adapter()
        params = OpenAICompletionRequestWithExtraBody(
            model="google/gemini-2.5-flash",
            prompt="hi",
            user="alice",
        )
        with caplog.at_level(logging.DEBUG, logger="ogx.providers.remote.inference.vertexai.vertexai"):
            await adapter.openai_completion(params)
        messages = " ".join(r.message for r in caplog.records)
        assert "user" in messages
        assert "text completion ignores" in messages

    async def test_sampling_params_mapped(self, make_completion_adapter):
        """Test that sampling params mapped."""
        adapter, fake_client = make_completion_adapter()
        params = OpenAICompletionRequestWithExtraBody(
            model="google/gemini-2.5-flash",
            prompt="hi",
            temperature=0.7,
            top_p=0.9,
            max_tokens=100,
            stop="END",
            seed=42,
        )
        await adapter.openai_completion(params)
        call_kwargs = fake_client.aio.models.generate_content.call_args.kwargs
        config = call_kwargs["config"]
        assert config.temperature == 0.7
        assert config.top_p == 0.9
        assert config.max_output_tokens == 100
        assert config.stop_sequences == ["END"]
        assert config.seed == 42

    async def test_logprobs_param_sets_response_logprobs(self, make_completion_adapter):
        """Test that logprobs param sets response logprobs."""
        adapter, fake_client = make_completion_adapter()
        params = OpenAICompletionRequestWithExtraBody(
            model="google/gemini-2.5-flash",
            prompt="hi",
            logprobs=True,
        )
        await adapter.openai_completion(params)
        call_kwargs = fake_client.aio.models.generate_content.call_args.kwargs
        config = call_kwargs["config"]
        assert config.response_logprobs is True

    async def test_logprobs_false_sets_response_logprobs_none(self, make_completion_adapter):
        """Test that logprobs false sets response logprobs none."""
        adapter, fake_client = make_completion_adapter()
        params = OpenAICompletionRequestWithExtraBody(
            model="google/gemini-2.5-flash",
            prompt="hi",
            logprobs=False,
        )
        await adapter.openai_completion(params)
        call_kwargs = fake_client.aio.models.generate_content.call_args.kwargs
        config = call_kwargs["config"]
        assert getattr(config, "response_logprobs", None) is None

    async def test_stream_raises_not_implemented_removed(self, monkeypatch):
        """Test that stream raises not implemented removed."""
        adapter = VertexAIInferenceAdapter(config=VertexAIConfig(project="p", location="l"))

        async def fake_stream(**kwargs):
            """Yield predefined fake stream chunks."""
            yield _make_fake_streaming_chunk("streamed")

        fake_client = SimpleNamespace(
            aio=SimpleNamespace(models=SimpleNamespace(generate_content_stream=AsyncMock(return_value=fake_stream())))
        )
        monkeypatch.setattr(adapter, "_get_provider_model_id", AsyncMock(return_value="gemini-2.5-flash"))
        monkeypatch.setattr(adapter, "_validate_model_allowed", lambda _: None)
        monkeypatch.setattr(adapter, "_get_client", lambda: fake_client)

        params = OpenAICompletionRequestWithExtraBody(model="google/gemini-2.5-flash", prompt="hi", stream=True)
        result = await adapter.openai_completion(params)
        chunks = [chunk async for chunk in cast(AsyncIterator[OpenAICompletion], result)]
        assert len(chunks) == 1
        assert isinstance(chunks[0], OpenAICompletion)

    async def test_stream_completion_with_string_prompt(self, monkeypatch):
        """Test that stream completion with string prompt."""
        adapter = VertexAIInferenceAdapter(config=VertexAIConfig(project="p", location="l"))

        chunk1_text = "Hello"
        chunk2_text = " world"

        async def fake_stream(**kwargs):
            """Yield predefined fake stream chunks."""
            for text in [chunk1_text, chunk2_text]:
                yield _make_fake_streaming_chunk(text)

        fake_client = SimpleNamespace(
            aio=SimpleNamespace(models=SimpleNamespace(generate_content_stream=AsyncMock(return_value=fake_stream())))
        )
        monkeypatch.setattr(adapter, "_get_provider_model_id", AsyncMock(return_value="gemini-2.5-flash"))
        monkeypatch.setattr(adapter, "_validate_model_allowed", lambda _: None)
        monkeypatch.setattr(adapter, "_get_client", lambda: fake_client)

        params = OpenAICompletionRequestWithExtraBody(model="google/gemini-2.5-flash", prompt="Say hello", stream=True)
        result = await adapter.openai_completion(params)

        chunks = [chunk async for chunk in cast(AsyncIterator[OpenAICompletion], result)]
        assert len(chunks) == 2
        assert all(isinstance(c, OpenAICompletion) for c in chunks)
        assert chunks[0].choices[0].text == chunk1_text
        assert chunks[1].choices[0].text == chunk2_text
        assert chunks[0].id == chunks[1].id

    async def test_stream_completion_rejects_list_prompt(self, monkeypatch):
        """Test that stream completion rejects list prompt."""
        adapter = VertexAIInferenceAdapter(config=VertexAIConfig(project="p", location="l"))
        fake_chunk = _make_fake_streaming_chunk("hi")
        fake_client = MagicMock()
        fake_client.aio.models.generate_content_stream = AsyncMock(return_value=_async_pager([fake_chunk]))
        monkeypatch.setattr(adapter, "_get_provider_model_id", AsyncMock(return_value="gemini-2.5-flash"))
        monkeypatch.setattr(adapter, "_validate_model_allowed", lambda _: None)
        monkeypatch.setattr(adapter, "_get_client", lambda: fake_client)
        params = OpenAICompletionRequestWithExtraBody(model="google/gemini-2.5-flash", prompt=["hello"], stream=True)
        result = await adapter.openai_completion(params)
        assert hasattr(result, "__aiter__")

    def _patch_streaming_dependencies(self, monkeypatch, adapter, fake_client):
        """Patch streaming dependencies for completion tests."""
        monkeypatch.setattr(adapter, "_get_provider_model_id", AsyncMock(return_value="gemini-2.5-flash"))
        monkeypatch.setattr(adapter, "_validate_model_allowed", lambda _: None)
        monkeypatch.setattr(adapter, "_get_client", lambda: fake_client)

    async def test_stream_multi_prompt_shared_completion_id(self, monkeypatch):
        """Test that stream multi prompt shared completion id."""
        adapter = VertexAIInferenceAdapter(config=VertexAIConfig(project="p", location="l"))
        chunk_a = _make_fake_streaming_chunk("response_a")
        chunk_b = _make_fake_streaming_chunk("response_b")
        fake_client = MagicMock()
        fake_client.aio.models.generate_content_stream = AsyncMock(
            side_effect=[_async_pager([chunk_a]), _async_pager([chunk_b])]
        )
        self._patch_streaming_dependencies(monkeypatch, adapter, fake_client)
        params = OpenAICompletionRequestWithExtraBody(model="google/gemini-2.5-flash", prompt=["a", "b"], stream=True)
        result = await adapter.openai_completion(params)
        chunks = [chunk async for chunk in cast(AsyncIterator[OpenAICompletion], result)]
        assert len(chunks) >= 2
        assert len({chunk.id for chunk in chunks}) == 1

    async def test_stream_multi_prompt_indices(self, monkeypatch):
        """Test that stream multi prompt indices."""
        adapter = VertexAIInferenceAdapter(config=VertexAIConfig(project="p", location="l"))
        chunk_a = _make_fake_streaming_chunk("response_a")
        chunk_b = _make_fake_streaming_chunk("response_b")
        fake_client = MagicMock()
        fake_client.aio.models.generate_content_stream = AsyncMock(
            side_effect=[_async_pager([chunk_a]), _async_pager([chunk_b])]
        )
        self._patch_streaming_dependencies(monkeypatch, adapter, fake_client)
        params = OpenAICompletionRequestWithExtraBody(model="google/gemini-2.5-flash", prompt=["a", "b"], stream=True)
        result = await adapter.openai_completion(params)
        chunks = [chunk async for chunk in cast(AsyncIterator[OpenAICompletion], result)]
        indices = [chunk.choices[0].index for chunk in chunks]
        assert 0 in indices
        assert 1 in indices

    async def test_stream_single_item_list_prompt(self, monkeypatch):
        """Test that stream single item list prompt."""
        adapter = VertexAIInferenceAdapter(config=VertexAIConfig(project="p", location="l"))
        chunk = _make_fake_streaming_chunk("response")
        fake_client = MagicMock()
        fake_client.aio.models.generate_content_stream = AsyncMock(return_value=_async_pager([chunk]))
        self._patch_streaming_dependencies(monkeypatch, adapter, fake_client)
        params = OpenAICompletionRequestWithExtraBody(model="google/gemini-2.5-flash", prompt=["hello"], stream=True)
        result = await adapter.openai_completion(params)
        chunks = [chunk async for chunk in cast(AsyncIterator[OpenAICompletion], result)]
        assert len(chunks) >= 1
        assert all(c.choices[0].index == 0 for c in chunks)

    async def test_echo_non_streaming_prepends_prompt(self, make_completion_adapter):
        """Test that echo non streaming prepends prompt."""
        adapter, _ = make_completion_adapter(fake_response=_make_fake_gemini_response(text="world"))
        params = OpenAICompletionRequestWithExtraBody(model="google/gemini-2.5-flash", prompt="hello", echo=True)
        result = await adapter.openai_completion(params)
        assert result.choices[0].text == "helloworld"

    async def test_echo_false_does_not_prepend(self, make_completion_adapter):
        """Test that echo false does not prepend."""
        adapter, _ = make_completion_adapter(fake_response=_make_fake_gemini_response(text="world"))
        params = OpenAICompletionRequestWithExtraBody(model="google/gemini-2.5-flash", prompt="hello")
        result = await adapter.openai_completion(params)
        assert result.choices[0].text == "world"

    async def test_echo_multi_prompt_non_streaming(self, make_completion_adapter):
        """Test that echo multi prompt non streaming."""
        adapter, _ = make_completion_adapter(fake_response=_make_fake_gemini_response(text="world"))
        params = OpenAICompletionRequestWithExtraBody(model="google/gemini-2.5-flash", prompt=["a", "b"], echo=True)
        result = await adapter.openai_completion(params)
        assert result.choices[0].text == "aworld"
        assert result.choices[1].text == "bworld"

    async def test_non_streaming_n_two_returns_two_choices(self, make_completion_adapter):
        """Test that non streaming n two returns two choices."""
        first = SimpleNamespace(text="first", function_call=None, thought=None)
        second = SimpleNamespace(text="second", function_call=None, thought=None)
        fake_response = SimpleNamespace(
            candidates=[
                SimpleNamespace(
                    content=SimpleNamespace(parts=[first]),
                    finish_reason="STOP",
                    index=0,
                    logprobs_result=None,
                ),
                SimpleNamespace(
                    content=SimpleNamespace(parts=[second]),
                    finish_reason="STOP",
                    index=1,
                    logprobs_result=None,
                ),
            ],
            usage_metadata=None,
        )
        adapter, _ = make_completion_adapter(fake_response=fake_response)
        params = OpenAICompletionRequestWithExtraBody(model="google/gemini-2.5-flash", prompt="hello", n=2)
        result = await adapter.openai_completion(params)
        assert len(result.choices) == 2
        assert result.choices[0].text == "first"
        assert result.choices[1].text == "second"
        assert [choice.index for choice in result.choices] == [0, 1]

    async def test_echo_streaming_emits_prompt_chunk(self, monkeypatch):
        """Test that echo streaming emits prompt chunk."""
        adapter = VertexAIInferenceAdapter(config=VertexAIConfig(project="p", location="l"))
        chunk = _make_fake_streaming_chunk("world")
        fake_client = MagicMock()
        fake_client.aio.models.generate_content_stream = AsyncMock(return_value=_async_pager([chunk]))
        self._patch_streaming_dependencies(monkeypatch, adapter, fake_client)
        params = OpenAICompletionRequestWithExtraBody(
            model="google/gemini-2.5-flash", prompt="hello", stream=True, echo=True
        )
        result = await adapter.openai_completion(params)
        chunks = [c async for c in cast(AsyncIterator[OpenAICompletion], result)]
        assert chunks[0].choices[0].text == "hello"


class TestCompletionModelExtra:
    async def test_non_streaming_model_extra_forwarded(self, make_completion_adapter):
        """Test that non streaming model extra forwarded."""
        adapter, fake_client = make_completion_adapter()
        params = OpenAICompletionRequestWithExtraBody.model_validate(
            {
                "model": "google/gemini-2.5-flash",
                "prompt": "Hello world",
                "safety_settings": "block_none",
            }
        )
        await adapter.openai_completion(params)
        call_kwargs = fake_client.aio.models.generate_content.call_args.kwargs
        config = call_kwargs["config"]
        assert config.safety_settings == "block_none"

    async def test_streaming_model_extra_forwarded(self, monkeypatch):
        """Test that streaming model extra forwarded."""
        adapter = VertexAIInferenceAdapter(config=VertexAIConfig(project="p", location="l"))

        async def fake_stream(**kwargs):
            """Yield predefined fake stream chunks."""
            yield _make_fake_streaming_chunk("streamed")

        fake_client = SimpleNamespace(
            aio=SimpleNamespace(models=SimpleNamespace(generate_content_stream=AsyncMock(return_value=fake_stream())))
        )
        monkeypatch.setattr(adapter, "_get_provider_model_id", AsyncMock(return_value="gemini-2.5-flash"))
        monkeypatch.setattr(adapter, "_validate_model_allowed", lambda _: None)
        monkeypatch.setattr(adapter, "_get_client", lambda: fake_client)

        params = OpenAICompletionRequestWithExtraBody.model_validate(
            {
                "model": "google/gemini-2.5-flash",
                "prompt": "Say hello",
                "stream": True,
                "safety_settings": "block_none",
            }
        )
        result = await adapter.openai_completion(params)
        chunks = [chunk async for chunk in cast(AsyncIterator[OpenAICompletion], result)]
        assert len(chunks) > 0

        call_kwargs = fake_client.aio.models.generate_content_stream.call_args.kwargs
        config = call_kwargs["config"]
        assert config.safety_settings == "block_none"

    async def test_no_model_extra_does_not_add_extra_fields(self, make_completion_adapter):
        """Test that no model extra does not add extra fields."""
        adapter, fake_client = make_completion_adapter()
        params = OpenAICompletionRequestWithExtraBody(
            model="google/gemini-2.5-flash",
            prompt="Hello world",
        )
        await adapter.openai_completion(params)
        call_kwargs = fake_client.aio.models.generate_content.call_args.kwargs
        config = call_kwargs["config"]
        assert not hasattr(config, "safety_settings")


class TestCompletionStreamOptions:
    def _patch_stream_completion(self, monkeypatch, adapter: VertexAIInferenceAdapter) -> dict[str, Any]:
        """Patch completion streaming hooks for telemetry tests."""
        fake_client = SimpleNamespace(
            aio=SimpleNamespace(models=SimpleNamespace(generate_content_stream=AsyncMock(return_value=None)))
        )
        stream_call_args: dict[str, Any] = {}

        async def _provider_model_id(_: str) -> str:
            """Return a fixed provider model identifier."""
            return "gemini-2.5-flash"

        async def _stream_completion(client, model_id, contents, config, model, stream_options=None):
            """Handle stream completion."""
            stream_call_args["stream_options"] = stream_options
            return _async_pager([])

        monkeypatch.setattr(adapter, "_get_provider_model_id", _provider_model_id)
        monkeypatch.setattr(adapter, "_validate_model_allowed", lambda _: None)
        monkeypatch.setattr(adapter, "_get_client", lambda: fake_client)
        monkeypatch.setattr(
            "ogx.providers.remote.inference.vertexai.vertexai.converters.convert_completion_prompt_to_contents",
            lambda _: [{"role": "user", "parts": [{"text": "ok"}]}],
        )
        monkeypatch.setattr(adapter, "_stream_completion", _stream_completion)

        return stream_call_args

    async def test_stream_options_injected_when_telemetry_active(self, monkeypatch):
        """Test that stream options injected when telemetry active."""
        adapter = VertexAIInferenceAdapter(config=VertexAIConfig(project="p", location="l"))
        stream_call_args = self._patch_stream_completion(monkeypatch, adapter)

        mock_span = MagicMock()
        mock_span.is_recording.return_value = True
        mock_trace = MagicMock()
        mock_trace.get_current_span.return_value = mock_span
        monkeypatch.setattr("opentelemetry.trace", mock_trace)

        params = OpenAICompletionRequestWithExtraBody(
            model="google/gemini-2.5-flash",
            prompt="Say hello",
            stream=True,
        )

        result = await adapter.openai_completion(params)
        async for _ in cast(AsyncIterator[OpenAICompletion], result):
            pass

        assert stream_call_args.get("stream_options") is not None
        assert stream_call_args["stream_options"].get("include_usage") is True

    async def test_stream_options_not_injected_when_telemetry_inactive(self, monkeypatch):
        """Test that stream options not injected when telemetry inactive."""
        adapter = VertexAIInferenceAdapter(config=VertexAIConfig(project="p", location="l"))
        stream_call_args = self._patch_stream_completion(monkeypatch, adapter)

        mock_span = MagicMock()
        mock_span.is_recording.return_value = False
        mock_trace = MagicMock()
        mock_trace.get_current_span.return_value = mock_span
        monkeypatch.setattr("opentelemetry.trace", mock_trace)

        params = OpenAICompletionRequestWithExtraBody(
            model="google/gemini-2.5-flash",
            prompt="Say hello",
            stream=True,
        )

        result = await adapter.openai_completion(params)
        async for _ in cast(AsyncIterator[OpenAICompletion], result):
            pass

        assert stream_call_args.get("stream_options") is None


class TestVertexAIModelSerialization:
    def test_model_dump_excludes_injected_extra_fields(self):
        """Test that model dump excludes injected extra fields."""
        vertex_config = VertexAIConfig(project="test-project", location="global")
        adapter = VertexAIInferenceAdapter.model_validate({"config": vertex_config, "model_store": object()})
        result = adapter.model_dump()
        assert "model_store" not in result
        assert "config" in result

    def test_model_dump_json_excludes_injected_extra_fields(self):
        """Test that model dump json excludes injected extra fields."""
        vertex_config = VertexAIConfig(project="test-project", location="global")
        adapter = VertexAIInferenceAdapter.model_validate({"config": vertex_config, "model_store": object()})
        result_json = adapter.model_dump_json()
        assert "model_store" not in result_json
        parsed = json.loads(result_json)
        assert "config" in parsed

    def test_model_dump_without_extra_fields_works_normally(self):
        """Test that model dump without extra fields works normally."""
        vertex_config = VertexAIConfig(project="test-project", location="global")
        adapter = VertexAIInferenceAdapter(config=vertex_config)
        result = adapter.model_dump()
        assert "config" in result
        assert "embedding_model_metadata" in result
