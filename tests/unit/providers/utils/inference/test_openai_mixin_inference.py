# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ogx_api import (
    OpenAIChatCompletionRequestWithExtraBody,
    OpenAICompletionRequestWithExtraBody,
    OpenAIEmbeddingsRequestWithExtraBody,
    OpenAIUserMessageParam,
)


class TestOpenAIMixinAllowedModelsInference:
    """Test cases for allowed_models enforcement during inference requests"""

    async def test_inference_with_allowed_models(self, mixin, mock_client_context):
        """Test that all inference methods succeed with allowed models"""
        mixin.config.allowed_models = ["gpt-4", "text-davinci-003", "text-embedding-ada-002"]

        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=MagicMock())
        mock_client.completions.create = AsyncMock(return_value=MagicMock())
        mock_embedding_response = MagicMock()
        mock_embedding_response.data = [MagicMock(embedding=[0.1, 0.2, 0.3])]
        mock_embedding_response.usage = MagicMock(prompt_tokens=5, total_tokens=5)
        mock_client.embeddings.create = AsyncMock(return_value=mock_embedding_response)

        with mock_client_context(mixin, mock_client):
            # Test chat completion
            await mixin.openai_chat_completion(
                OpenAIChatCompletionRequestWithExtraBody(
                    model="gpt-4", messages=[OpenAIUserMessageParam(role="user", content="Hello")]
                )
            )
            mock_client.chat.completions.create.assert_called_once()

            # Test completion
            await mixin.openai_completion(
                OpenAICompletionRequestWithExtraBody(model="text-davinci-003", prompt="Hello")
            )
            mock_client.completions.create.assert_called_once()

            # Test embeddings
            await mixin.openai_embeddings(
                OpenAIEmbeddingsRequestWithExtraBody(model="text-embedding-ada-002", input="test text")
            )
            mock_client.embeddings.create.assert_called_once()

    async def test_inference_with_disallowed_models(self, mixin, mock_client_context):
        """Test that all inference methods fail with disallowed models"""
        mixin.config.allowed_models = ["gpt-4"]

        mock_client = MagicMock()

        with mock_client_context(mixin, mock_client):
            # Test chat completion with disallowed model
            with pytest.raises(ValueError, match="Model 'gpt-4-turbo' is not in the allowed models list"):
                await mixin.openai_chat_completion(
                    OpenAIChatCompletionRequestWithExtraBody(
                        model="gpt-4-turbo", messages=[OpenAIUserMessageParam(role="user", content="Hello")]
                    )
                )

            # Test completion with disallowed model
            with pytest.raises(ValueError, match="Model 'text-davinci-002' is not in the allowed models list"):
                await mixin.openai_completion(
                    OpenAICompletionRequestWithExtraBody(model="text-davinci-002", prompt="Hello")
                )

            # Test embeddings with disallowed model
            with pytest.raises(ValueError, match="Model 'text-embedding-3-large' is not in the allowed models list"):
                await mixin.openai_embeddings(
                    OpenAIEmbeddingsRequestWithExtraBody(model="text-embedding-3-large", input="test text")
                )

            mock_client.chat.completions.create.assert_not_called()
            mock_client.completions.create.assert_not_called()
            mock_client.embeddings.create.assert_not_called()

    async def test_inference_with_no_restrictions(self, mixin, mock_client_context):
        """Test that inference succeeds when allowed_models is None or empty list blocks all"""
        # Test with None (no restrictions)
        assert mixin.config.allowed_models is None
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=MagicMock())

        with mock_client_context(mixin, mock_client):
            await mixin.openai_chat_completion(
                OpenAIChatCompletionRequestWithExtraBody(
                    model="any-model", messages=[OpenAIUserMessageParam(role="user", content="Hello")]
                )
            )
            mock_client.chat.completions.create.assert_called_once()

        # Test with empty list (blocks all models)
        mixin.config.allowed_models = []
        with mock_client_context(mixin, mock_client):
            with pytest.raises(ValueError, match="Model 'gpt-4' is not in the allowed models list"):
                await mixin.openai_chat_completion(
                    OpenAIChatCompletionRequestWithExtraBody(
                        model="gpt-4", messages=[OpenAIUserMessageParam(role="user", content="Hello")]
                    )
                )


class TestOpenAIMixinStreamOptionsInjection:
    """Test cases for automatic stream_options injection when telemetry is active"""

    async def test_chat_completion_injects_stream_options_when_telemetry_active(self, mixin, mock_client_context):
        """Test that stream_options is injected for streaming chat completion when telemetry is active"""
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=MagicMock())

        # Mock OpenTelemetry span as recording
        mock_span = MagicMock()
        mock_span.is_recording.return_value = True

        with mock_client_context(mixin, mock_client):
            with patch("opentelemetry.trace.get_current_span", return_value=mock_span):
                await mixin.openai_chat_completion(
                    OpenAIChatCompletionRequestWithExtraBody(
                        model="gpt-4", messages=[OpenAIUserMessageParam(role="user", content="Hello")], stream=True
                    )
                )

                mock_client.chat.completions.create.assert_called_once()
                call_kwargs = mock_client.chat.completions.create.call_args[1]
                assert call_kwargs["stream_options"] == {"include_usage": True}

    async def test_chat_completion_preserves_existing_stream_options(self, mixin, mock_client_context):
        """Test that existing stream_options are preserved with include_usage added"""
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=MagicMock())

        mock_span = MagicMock()
        mock_span.is_recording.return_value = True

        with mock_client_context(mixin, mock_client):
            with patch("opentelemetry.trace.get_current_span", return_value=mock_span):
                await mixin.openai_chat_completion(
                    OpenAIChatCompletionRequestWithExtraBody(
                        model="gpt-4",
                        messages=[OpenAIUserMessageParam(role="user", content="Hello")],
                        stream=True,
                        stream_options={"other_option": True},
                    )
                )

                call_kwargs = mock_client.chat.completions.create.call_args[1]
                assert call_kwargs["stream_options"] == {"other_option": True, "include_usage": True}

    async def test_chat_completion_no_injection_when_telemetry_inactive(self, mixin, mock_client_context):
        """Test that stream_options is NOT injected when telemetry is inactive"""
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=MagicMock())

        # Mock OpenTelemetry span as not recording
        mock_span = MagicMock()
        mock_span.is_recording.return_value = False

        with mock_client_context(mixin, mock_client):
            with patch("opentelemetry.trace.get_current_span", return_value=mock_span):
                await mixin.openai_chat_completion(
                    OpenAIChatCompletionRequestWithExtraBody(
                        model="gpt-4", messages=[OpenAIUserMessageParam(role="user", content="Hello")], stream=True
                    )
                )

                call_kwargs = mock_client.chat.completions.create.call_args[1]
                assert "stream_options" not in call_kwargs or call_kwargs["stream_options"] is None

    async def test_chat_completion_no_injection_when_not_streaming(self, mixin, mock_client_context):
        """Test that stream_options is NOT injected for non-streaming requests"""
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=MagicMock())

        mock_span = MagicMock()
        mock_span.is_recording.return_value = True

        with mock_client_context(mixin, mock_client):
            with patch("opentelemetry.trace.get_current_span", return_value=mock_span):
                await mixin.openai_chat_completion(
                    OpenAIChatCompletionRequestWithExtraBody(
                        model="gpt-4", messages=[OpenAIUserMessageParam(role="user", content="Hello")], stream=False
                    )
                )

                call_kwargs = mock_client.chat.completions.create.call_args[1]
                assert "stream_options" not in call_kwargs or call_kwargs["stream_options"] is None

    async def test_completion_injects_stream_options_when_telemetry_active(self, mixin, mock_client_context):
        """Test that stream_options is injected for streaming completion when telemetry is active"""
        mock_client = MagicMock()
        mock_client.completions.create = AsyncMock(return_value=MagicMock())

        mock_span = MagicMock()
        mock_span.is_recording.return_value = True

        with mock_client_context(mixin, mock_client):
            with patch("opentelemetry.trace.get_current_span", return_value=mock_span):
                await mixin.openai_completion(
                    OpenAICompletionRequestWithExtraBody(model="text-davinci-003", prompt="Hello", stream=True)
                )

                mock_client.completions.create.assert_called_once()
                call_kwargs = mock_client.completions.create.call_args[1]
                assert call_kwargs["stream_options"] == {"include_usage": True}

    async def test_completion_no_injection_when_telemetry_inactive(self, mixin, mock_client_context):
        """Test that stream_options is NOT injected for completion when telemetry is inactive"""
        mock_client = MagicMock()
        mock_client.completions.create = AsyncMock(return_value=MagicMock())

        mock_span = MagicMock()
        mock_span.is_recording.return_value = False

        with mock_client_context(mixin, mock_client):
            with patch("opentelemetry.trace.get_current_span", return_value=mock_span):
                await mixin.openai_completion(
                    OpenAICompletionRequestWithExtraBody(model="text-davinci-003", prompt="Hello", stream=True)
                )

                call_kwargs = mock_client.completions.create.call_args[1]
                assert "stream_options" not in call_kwargs or call_kwargs["stream_options"] is None

    async def test_params_not_mutated(self, mixin, mock_client_context):
        """Test that original params object is not mutated when stream_options is injected"""
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=MagicMock())

        mock_span = MagicMock()
        mock_span.is_recording.return_value = True

        original_params = OpenAIChatCompletionRequestWithExtraBody(
            model="gpt-4", messages=[OpenAIUserMessageParam(role="user", content="Hello")], stream=True
        )

        with mock_client_context(mixin, mock_client):
            with patch("opentelemetry.trace.get_current_span", return_value=mock_span):
                await mixin.openai_chat_completion(original_params)

                # Original params should not be modified
                assert original_params.stream_options is None

    async def test_chat_completion_overrides_include_usage_false(self, mixin, mock_client_context):
        """Test that include_usage=False is overridden when telemetry is active"""
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=MagicMock())

        mock_span = MagicMock()
        mock_span.is_recording.return_value = True

        with mock_client_context(mixin, mock_client):
            with patch("opentelemetry.trace.get_current_span", return_value=mock_span):
                await mixin.openai_chat_completion(
                    OpenAIChatCompletionRequestWithExtraBody(
                        model="gpt-4",
                        messages=[OpenAIUserMessageParam(role="user", content="Hello")],
                        stream=True,
                        stream_options={"include_usage": False},
                    )
                )

                call_kwargs = mock_client.chat.completions.create.call_args[1]
                # Telemetry must override False to ensure complete metrics
                assert call_kwargs["stream_options"]["include_usage"] is True

    async def test_no_injection_when_provider_doesnt_support_stream_options(self, mixin, mock_client_context):
        """Test that stream_options is NOT injected when provider doesn't support it"""
        # Set supports_stream_options to False (like Ollama/vLLM)
        mixin.supports_stream_options = False

        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=MagicMock())

        # Mock OpenTelemetry span as recording (telemetry is active)
        mock_span = MagicMock()
        mock_span.is_recording.return_value = True

        with mock_client_context(mixin, mock_client):
            with patch("opentelemetry.trace.get_current_span", return_value=mock_span):
                await mixin.openai_chat_completion(
                    OpenAIChatCompletionRequestWithExtraBody(
                        model="gpt-4", messages=[OpenAIUserMessageParam(role="user", content="Hello")], stream=True
                    )
                )

                call_kwargs = mock_client.chat.completions.create.call_args[1]
                # Should NOT inject stream_options even though telemetry is active
                assert "stream_options" not in call_kwargs or call_kwargs["stream_options"] is None

    async def test_completion_no_injection_when_provider_doesnt_support_stream_options(
        self, mixin, mock_client_context
    ):
        """Test that stream_options is NOT injected for completion when provider doesn't support it"""
        # Set supports_stream_options to False (like Ollama/vLLM)
        mixin.supports_stream_options = False

        mock_client = MagicMock()
        mock_client.completions.create = AsyncMock(return_value=MagicMock())

        # Mock OpenTelemetry span as recording (telemetry is active)
        mock_span = MagicMock()
        mock_span.is_recording.return_value = True

        with mock_client_context(mixin, mock_client):
            with patch("opentelemetry.trace.get_current_span", return_value=mock_span):
                await mixin.openai_completion(
                    OpenAICompletionRequestWithExtraBody(model="text-davinci-003", prompt="Hello", stream=True)
                )

                call_kwargs = mock_client.completions.create.call_args[1]
                # Should NOT inject stream_options even though telemetry is active
                assert "stream_options" not in call_kwargs or call_kwargs["stream_options"] is None


class TestOpenAIMixinSafetyIdentifierPassing:
    """Test cases for safety_identifier parameter passing to OpenAI API"""

    async def test_chat_completion_passes_safety_identifier(self, mixin, mock_client_context):
        """Test that safety_identifier is passed to OpenAI chat completions API"""
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=MagicMock())

        with mock_client_context(mixin, mock_client):
            await mixin.openai_chat_completion(
                OpenAIChatCompletionRequestWithExtraBody(
                    model="gpt-4",
                    messages=[OpenAIUserMessageParam(role="user", content="Hello")],
                    safety_identifier="user-123-hashed",
                )
            )

            mock_client.chat.completions.create.assert_called_once()
            call_kwargs = mock_client.chat.completions.create.call_args[1]
            assert call_kwargs["safety_identifier"] == "user-123-hashed"

    async def test_chat_completion_with_top_p(self, mixin, mock_client_context):
        """Test that top_p is properly passed to the OpenAI client"""
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=MagicMock())

        top_p_value = 0.9

        with mock_client_context(mixin, mock_client):
            await mixin.openai_chat_completion(
                OpenAIChatCompletionRequestWithExtraBody(
                    model="gpt-4",
                    messages=[OpenAIUserMessageParam(role="user", content="Hello")],
                    top_p=top_p_value,
                )
            )

            mock_client.chat.completions.create.assert_called_once()
            call_kwargs = mock_client.chat.completions.create.call_args[1]
            assert call_kwargs["top_p"] == top_p_value


class TestOpenAIMixinPromptCacheKey:
    """Test cases for prompt_cache_key parameter propagation"""

    async def test_chat_completion_with_prompt_cache_key(self, mixin, mock_client_context):
        """Test that prompt_cache_key is properly passed to the OpenAI client"""
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=MagicMock())

        cache_key = "test-cache-key-123"

        with mock_client_context(mixin, mock_client):
            await mixin.openai_chat_completion(
                OpenAIChatCompletionRequestWithExtraBody(
                    model="gpt-4",
                    messages=[OpenAIUserMessageParam(role="user", content="Hello")],
                    prompt_cache_key=cache_key,
                )
            )

            mock_client.chat.completions.create.assert_called_once()
            call_kwargs = mock_client.chat.completions.create.call_args[1]
            assert call_kwargs["prompt_cache_key"] == cache_key


class TestOpenAIMixinServiceTier:
    """Test cases for service_tier parameter in OpenAIMixin"""

    async def test_chat_completion_passes_service_tier_to_openai(self, mixin, mock_client_context):
        """Test that service_tier parameter is passed to OpenAI client for chat completion"""
        from ogx_api.inference import ServiceTier

        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=MagicMock())

        with mock_client_context(mixin, mock_client):
            await mixin.openai_chat_completion(
                OpenAIChatCompletionRequestWithExtraBody(
                    model="gpt-4",
                    messages=[OpenAIUserMessageParam(role="user", content="Hello")],
                    service_tier=ServiceTier.priority,
                )
            )

            mock_client.chat.completions.create.assert_called_once()
            call_kwargs = mock_client.chat.completions.create.call_args[1]
            assert call_kwargs["service_tier"] == ServiceTier.priority


class TestOpenAIMixinTopLogprobs:
    """Test cases for top_logprobs parameter in chat completion requests"""

    async def test_chat_completion_with_top_logprobs_value_5(self, mixin, mock_client_context):
        """Test that top_logprobs=5 is properly passed to the OpenAI client"""
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=MagicMock())

        with mock_client_context(mixin, mock_client):
            await mixin.openai_chat_completion(
                OpenAIChatCompletionRequestWithExtraBody(
                    model="gpt-4",
                    messages=[OpenAIUserMessageParam(role="user", content="Hello")],
                    top_logprobs=5,
                )
            )

            mock_client.chat.completions.create.assert_called_once()
            call_kwargs = mock_client.chat.completions.create.call_args[1]
            assert call_kwargs["top_logprobs"] == 5

    async def test_chat_completion_with_top_logprobs_boundary_min(self, mixin, mock_client_context):
        """Test that top_logprobs=0 (minimum) is properly passed to the OpenAI client"""
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=MagicMock())

        with mock_client_context(mixin, mock_client):
            await mixin.openai_chat_completion(
                OpenAIChatCompletionRequestWithExtraBody(
                    model="gpt-4",
                    messages=[OpenAIUserMessageParam(role="user", content="Hello")],
                    top_logprobs=0,
                )
            )

            mock_client.chat.completions.create.assert_called_once()
            call_kwargs = mock_client.chat.completions.create.call_args[1]
            assert call_kwargs["top_logprobs"] == 0

    async def test_chat_completion_with_top_logprobs_boundary_max(self, mixin, mock_client_context):
        """Test that top_logprobs=20 (maximum) is properly passed to the OpenAI client"""
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=MagicMock())

        with mock_client_context(mixin, mock_client):
            await mixin.openai_chat_completion(
                OpenAIChatCompletionRequestWithExtraBody(
                    model="gpt-4",
                    messages=[OpenAIUserMessageParam(role="user", content="Hello")],
                    top_logprobs=20,
                )
            )

            mock_client.chat.completions.create.assert_called_once()
            call_kwargs = mock_client.chat.completions.create.call_args[1]
            assert call_kwargs["top_logprobs"] == 20


class TestOpenAIMixinUserProvidedStreamOptions:
    """Test cases for user-provided stream_options parameter handling"""

    async def test_user_stream_options_passed_through_when_telemetry_inactive(self, mixin, mock_client_context):
        """Test that user-provided stream_options are passed through unchanged when telemetry is inactive"""
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=MagicMock())

        mock_span = MagicMock()
        mock_span.is_recording.return_value = False

        # OpenAI stream_options supports include_usage (bool) and include_obfuscation (bool)
        # Using dict[str, Any] allows for future extensions and provider-specific options
        user_stream_options = {"include_obfuscation": True, "custom_field": 123}

        with mock_client_context(mixin, mock_client):
            with patch("opentelemetry.trace.get_current_span", return_value=mock_span):
                await mixin.openai_chat_completion(
                    OpenAIChatCompletionRequestWithExtraBody(
                        model="gpt-4",
                        messages=[OpenAIUserMessageParam(role="user", content="Hello")],
                        stream=True,
                        stream_options=user_stream_options,
                    )
                )

                call_kwargs = mock_client.chat.completions.create.call_args[1]
                # User's stream_options should be passed through unchanged
                assert call_kwargs["stream_options"] == user_stream_options

    async def test_user_stream_options_include_usage_false_overridden_by_telemetry(self, mixin, mock_client_context):
        """Test that include_usage=False is overridden to True when telemetry is active"""
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=MagicMock())

        mock_span = MagicMock()
        mock_span.is_recording.return_value = True

        with mock_client_context(mixin, mock_client):
            with patch("opentelemetry.trace.get_current_span", return_value=mock_span):
                await mixin.openai_chat_completion(
                    OpenAIChatCompletionRequestWithExtraBody(
                        model="gpt-4",
                        messages=[OpenAIUserMessageParam(role="user", content="Hello")],
                        stream=True,
                        stream_options={"include_usage": False, "other_option": True},
                    )
                )

                call_kwargs = mock_client.chat.completions.create.call_args[1]
                # Telemetry must override include_usage to True
                assert call_kwargs["stream_options"]["include_usage"] is True
                # Other options should be preserved
                assert call_kwargs["stream_options"]["other_option"] is True
