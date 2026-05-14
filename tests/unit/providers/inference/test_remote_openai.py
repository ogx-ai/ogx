# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import pytest

from ogx.providers.remote.inference.openai.config import OpenAIConfig
from ogx.providers.remote.inference.openai.openai import (
    _MODEL_MAX_OUTPUT_TOKENS,
    _WARNED_MODELS,
    OpenAIInferenceAdapter,
)
from ogx_api import (
    OpenAIChatCompletion,
    OpenAIChatCompletionRequestWithExtraBody,
    OpenAIChatCompletionResponseMessage,
    OpenAIChoice,
)


@pytest.fixture
def mock_openai_response():
    return OpenAIChatCompletion(
        id="chatcmpl-abc123",
        created=1,
        model="gpt-4o-mini",
        choices=[
            OpenAIChoice(
                message=OpenAIChatCompletionResponseMessage(content="hello"),
                finish_reason="stop",
                index=0,
            )
        ],
    )


@pytest.fixture(autouse=True)
def _clear_warned_models():
    _WARNED_MODELS.clear()
    yield
    _WARNED_MODELS.clear()


def _make_adapter():
    config = OpenAIConfig(api_key="fake-key")
    adapter = OpenAIInferenceAdapter(config=config)
    adapter.model_store = AsyncMock()
    return adapter


class TestOpenAIMaxTokensClamping:
    async def test_clamps_when_request_exceeds_model_limit(self, mock_openai_response):
        adapter = _make_adapter()

        with patch.object(OpenAIInferenceAdapter, "client", new_callable=PropertyMock) as mock_client_prop:
            mock_client = MagicMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_openai_response)
            mock_client_prop.return_value = mock_client

            params = OpenAIChatCompletionRequestWithExtraBody(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "hi"}],
                stream=False,
                max_tokens=32000,
            )
            await adapter.openai_chat_completion(params)

            call_kwargs = mock_client.chat.completions.create.call_args.kwargs
            assert call_kwargs["max_tokens"] == 16384

    async def test_keeps_lower_request_value(self, mock_openai_response):
        adapter = _make_adapter()

        with patch.object(OpenAIInferenceAdapter, "client", new_callable=PropertyMock) as mock_client_prop:
            mock_client = MagicMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_openai_response)
            mock_client_prop.return_value = mock_client

            params = OpenAIChatCompletionRequestWithExtraBody(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "hi"}],
                stream=False,
                max_tokens=1000,
            )
            await adapter.openai_chat_completion(params)

            call_kwargs = mock_client.chat.completions.create.call_args.kwargs
            assert call_kwargs["max_tokens"] == 1000

    async def test_no_clamping_when_max_tokens_is_none(self, mock_openai_response):
        adapter = _make_adapter()

        with patch.object(OpenAIInferenceAdapter, "client", new_callable=PropertyMock) as mock_client_prop:
            mock_client = MagicMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_openai_response)
            mock_client_prop.return_value = mock_client

            params = OpenAIChatCompletionRequestWithExtraBody(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "hi"}],
                stream=False,
            )
            await adapter.openai_chat_completion(params)

            call_kwargs = mock_client.chat.completions.create.call_args.kwargs
            assert call_kwargs.get("max_tokens") is None

    async def test_does_not_mutate_original_params(self, mock_openai_response):
        adapter = _make_adapter()

        with patch.object(OpenAIInferenceAdapter, "client", new_callable=PropertyMock) as mock_client_prop:
            mock_client = MagicMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_openai_response)
            mock_client_prop.return_value = mock_client

            params = OpenAIChatCompletionRequestWithExtraBody(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "hi"}],
                stream=False,
                max_tokens=32000,
            )
            await adapter.openai_chat_completion(params)

            assert params.max_tokens == 32000

    async def test_different_models_have_different_limits(self, mock_openai_response):
        adapter = _make_adapter()

        with patch.object(OpenAIInferenceAdapter, "client", new_callable=PropertyMock) as mock_client_prop:
            mock_client = MagicMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_openai_response)
            mock_client_prop.return_value = mock_client

            # gpt-4-turbo has a 4096 limit
            params = OpenAIChatCompletionRequestWithExtraBody(
                model="gpt-4-turbo",
                messages=[{"role": "user", "content": "hi"}],
                stream=False,
                max_tokens=32000,
            )
            await adapter.openai_chat_completion(params)

            call_kwargs = mock_client.chat.completions.create.call_args.kwargs
            assert call_kwargs["max_tokens"] == 4096

    async def test_no_clamping_for_unknown_model(self, mock_openai_response):
        adapter = _make_adapter()

        with patch.object(OpenAIInferenceAdapter, "client", new_callable=PropertyMock) as mock_client_prop:
            mock_client = MagicMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_openai_response)
            mock_client_prop.return_value = mock_client

            params = OpenAIChatCompletionRequestWithExtraBody(
                model="some-future-model",
                messages=[{"role": "user", "content": "hi"}],
                stream=False,
                max_tokens=32000,
            )
            await adapter.openai_chat_completion(params)

            call_kwargs = mock_client.chat.completions.create.call_args.kwargs
            assert call_kwargs["max_tokens"] == 32000

    async def test_dated_snapshot_model_uses_base_limit(self, mock_openai_response):
        adapter = _make_adapter()

        with patch.object(OpenAIInferenceAdapter, "client", new_callable=PropertyMock) as mock_client_prop:
            mock_client = MagicMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_openai_response)
            mock_client_prop.return_value = mock_client

            params = OpenAIChatCompletionRequestWithExtraBody(
                model="gpt-4o-2024-08-06",
                messages=[{"role": "user", "content": "hi"}],
                stream=False,
                max_tokens=32000,
            )
            await adapter.openai_chat_completion(params)

            call_kwargs = mock_client.chat.completions.create.call_args.kwargs
            assert call_kwargs["max_tokens"] == 16384

    async def test_clamps_max_completion_tokens_when_request_exceeds_model_limit(self, mock_openai_response):
        adapter = _make_adapter()

        with patch.object(OpenAIInferenceAdapter, "client", new_callable=PropertyMock) as mock_client_prop:
            mock_client = MagicMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_openai_response)
            mock_client_prop.return_value = mock_client

            params = OpenAIChatCompletionRequestWithExtraBody(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "hi"}],
                stream=False,
                max_completion_tokens=32000,
            )
            await adapter.openai_chat_completion(params)

            call_kwargs = mock_client.chat.completions.create.call_args.kwargs
            assert call_kwargs["max_completion_tokens"] == 16384

    async def test_clamps_both_max_token_fields_when_both_exceed_model_limit(self, mock_openai_response):
        adapter = _make_adapter()

        with patch.object(OpenAIInferenceAdapter, "client", new_callable=PropertyMock) as mock_client_prop:
            mock_client = MagicMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_openai_response)
            mock_client_prop.return_value = mock_client

            params = OpenAIChatCompletionRequestWithExtraBody(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "hi"}],
                stream=False,
                max_tokens=32000,
                max_completion_tokens=32000,
            )
            await adapter.openai_chat_completion(params)

            call_kwargs = mock_client.chat.completions.create.call_args.kwargs
            assert call_kwargs["max_tokens"] == 16384
            assert call_kwargs["max_completion_tokens"] == 16384


class TestOpenAIModelMetadata:
    def test_construct_model_includes_max_output_tokens(self):
        adapter = _make_adapter()
        adapter.__provider_id__ = "openai"

        model = adapter.construct_model_from_identifier("gpt-4o-mini")
        assert model.metadata["max_output_tokens"] == 16384

    def test_construct_model_unknown_has_no_max_output_tokens(self):
        adapter = _make_adapter()
        adapter.__provider_id__ = "openai"

        model = adapter.construct_model_from_identifier("some-future-model")
        assert "max_output_tokens" not in model.metadata

    def test_construct_model_embedding_unchanged(self):
        adapter = _make_adapter()
        adapter.__provider_id__ = "openai"

        model = adapter.construct_model_from_identifier("text-embedding-3-small")
        assert model.model_type.value == "embedding"
        assert model.metadata["embedding_dimension"] == 1536


class TestOpenAIMaxOutputTokensWarning:
    def test_warns_once_for_unknown_model(self, caplog):
        adapter = _make_adapter()

        with caplog.at_level("WARNING"):
            result1 = adapter._get_max_output_tokens("brand-new-model")
            result2 = adapter._get_max_output_tokens("brand-new-model")

        assert result1 is None
        assert result2 is None
        warning_count = sum(1 for r in caplog.records if "brand-new-model" in r.message)
        assert warning_count == 1

    def test_all_known_models_have_limits(self):
        adapter = _make_adapter()
        for model_id, expected_limit in _MODEL_MAX_OUTPUT_TOKENS.items():
            assert adapter._get_max_output_tokens(model_id) == expected_limit

    def test_prefix_matching_prefers_more_specific_model(self):
        adapter = _make_adapter()
        assert adapter._get_max_output_tokens("o1-mini-2024-09-12") == 65536
