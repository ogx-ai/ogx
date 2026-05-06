# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import time

import pytest

from .helpers import skip_if_provider_is_vertexai
from .streaming_assertions import StreamingValidator


@pytest.mark.integration
class TestOpenAIResponses:
    """Integration tests for the OpenAI responses API."""

    def _invalid_base64_image_input(self):
        return [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "What is in this image?"},
                    {
                        "type": "input_image",
                        "image_url": "data:image/png;base64,not_valid_base64_data!!!",
                    },
                ],
            }
        ]

    def test_openai_response_with_max_output_tokens(self, openai_client, text_model_id):
        """Test OpenAI response with max_output_tokens parameter."""
        if text_model_id.startswith("watsonx/"):
            pytest.skip("WatsonX does not support max_output_tokens parameter")
        response = openai_client.responses.create(
            model=text_model_id,
            input=[{"role": "user", "content": "What are the 5 Ds of dodgeball?"}],
            max_output_tokens=100,
        )

        assert response.id.startswith("resp_")
        assert len(response.output_text.strip()) > 0
        assert response.max_output_tokens == 100

    def test_openai_response_with_small_max_output_tokens(self, openai_client, client_with_models, text_model_id):
        """Test response with very small max_output_tokens to trigger potential truncation."""
        if text_model_id.startswith("watsonx/"):
            pytest.skip("WatsonX does not support max_output_tokens parameter")
        skip_if_provider_is_vertexai(
            client_with_models, text_model_id, "does not strictly respect very small max_output_tokens limits"
        )
        response = openai_client.responses.create(
            model=text_model_id,
            input=[
                {
                    "role": "user",
                    "content": "Write a detailed essay about the history of artificial intelligence, covering the past 70 years.",
                }
            ],
            max_output_tokens=20,
        )

        assert response.id.startswith("resp_")
        assert response.max_output_tokens == 20
        assert len(response.output_text.strip()) > 0

        # With such a small token limit, the response might be incomplete
        # Note: The status might be 'incomplete' depending on provider implementation
        if response.usage is not None and response.usage.output_tokens > 0:
            # Allow some tolerance for provider differences
            assert response.usage.output_tokens <= 25, (
                f"Output tokens ({response.usage.output_tokens}) should respect max_output_tokens (20) "
            )

    def test_openai_response_max_output_tokens_below_minimum(self, openai_client, text_model_id):
        """Test that max_output_tokens below minimum (< 16) is rejected."""
        with pytest.raises(Exception) as exc_info:
            openai_client.responses.create(
                model=text_model_id,
                input=[{"role": "user", "content": "Hello"}],
                max_output_tokens=15,
            )

        # Should get a validation error
        error_message = str(exc_info.value).lower()
        assert "validation" in error_message or "invalid" in error_message or "16" in error_message

    def test_openai_response_streaming_failed_error_code_is_spec_compliant(self, openai_client, text_model_id):
        """Verify streaming failures produce a spec-compliant error code."""
        stream = openai_client.responses.create(
            model=text_model_id,
            input="Hello",
            stream=True,
            truncation="auto",
        )

        chunks = list(stream)
        validator = StreamingValidator(chunks)
        validator.assert_basic_event_sequence()

        failed_events = [e for e in chunks if e.type == "response.failed"]
        assert len(failed_events) == 1, f"Expected exactly one response.failed event, got {len(failed_events)}"

        validator.validate_event_structure()

    def test_openai_response_streaming_invalid_base64_image_failure_code_is_spec_compliant(
        self, openai_client, text_model_id
    ):
        """Verify invalid base64 image input becomes response.failed with a spec-compliant error code."""
        if text_model_id.startswith("watsonx/"):
            pytest.skip("WatsonX text model does not support image inputs")
        if text_model_id.startswith("ollama/"):
            # In some replay environments, Ollama models may not be exposed via `models.list()`.
            available_model_ids = {m.id for m in openai_client.models.list()}
            if text_model_id not in available_model_ids:
                pytest.skip(f"Model {text_model_id} not available in this environment")

        stream = openai_client.responses.create(
            model=text_model_id,
            input=self._invalid_base64_image_input(),
            stream=True,
        )

        chunks = list(stream)
        validator = StreamingValidator(chunks)
        validator.assert_basic_event_sequence()

        failed_events = [e for e in chunks if e.type == "response.failed"]
        assert len(failed_events) == 1, f"Expected exactly one response.failed event, got {len(failed_events)}"

        error = failed_events[0].response.error
        assert error is not None

        validator.validate_event_structure()

        if text_model_id.startswith("openai/"):
            assert error.code == "invalid_base64_image"

        if text_model_id.startswith("ollama/"):
            assert error.code in {"invalid_base64_image", "server_error"}

    def test_openai_response_with_prompt_cache_key(self, openai_client, text_model_id):
        """Test OpenAI response with prompt_cache_key parameter."""
        cache_key = "test-cache-key-001"
        response = openai_client.responses.create(
            model=text_model_id,
            input=[{"role": "user", "content": "What is the capital of France?"}],
            prompt_cache_key=cache_key,
        )

        assert response.id.startswith("resp_")
        assert len(response.output_text.strip()) > 0
        assert response.prompt_cache_key == cache_key

    def test_openai_response_with_prompt_cache_key_streaming(self, openai_client, text_model_id):
        """Test OpenAI response with prompt_cache_key in streaming mode."""
        cache_key = "test-cache-key-streaming-001"
        stream = openai_client.responses.create(
            model=text_model_id,
            input=[{"role": "user", "content": "What is the capital of Germany?"}],
            prompt_cache_key=cache_key,
            stream=True,
        )

        chunks = list(stream)
        validator = StreamingValidator(chunks)
        validator.assert_basic_event_sequence()
        validator.validate_event_structure()

        # Verify cache key is in the created event
        created_events = [e for e in chunks if e.type == "response.created"]
        assert len(created_events) == 1
        assert created_events[0].response.prompt_cache_key == cache_key

        # Verify cache key is in the completed event
        completed_events = [e for e in chunks if e.type == "response.completed"]
        assert len(completed_events) == 1
        assert completed_events[0].response.prompt_cache_key == cache_key

    def test_openai_response_with_prompt_cache_key_and_previous_response(self, openai_client, text_model_id):
        """Test that prompt_cache_key works correctly with previous_response_id."""
        cache_key = "conversation-cache-001"

        # Create first response
        response1 = openai_client.responses.create(
            model=text_model_id,
            input=[{"role": "user", "content": "What is 2+2?"}],
            prompt_cache_key=cache_key,
        )

        assert response1.id.startswith("resp_")
        assert response1.prompt_cache_key == cache_key

        # Create second response referencing the first one with the same cache key
        response2 = openai_client.responses.create(
            model=text_model_id,
            input=[{"role": "user", "content": "What is 3+3?"}],
            previous_response_id=response1.id,
            prompt_cache_key=cache_key,
        )

        assert response2.id.startswith("resp_")
        assert response2.prompt_cache_key == cache_key
        assert len(response2.output_text.strip()) > 0

    def test_openai_response_with_safety_identifier(self, openai_client, text_model_id):
        """Test OpenAI response with safety_identifier parameter."""
        safety_id = "user-123-hashed"
        response = openai_client.responses.create(
            model=text_model_id,
            input=[{"role": "user", "content": "What is the capital of Spain?"}],
            safety_identifier=safety_id,
        )

        assert response.id.startswith("resp_")
        assert len(response.output_text.strip()) > 0
        assert response.safety_identifier == safety_id

    def test_openai_response_with_safety_identifier_streaming(self, openai_client, text_model_id):
        """Test OpenAI response with safety_identifier in streaming mode."""
        safety_id = "user-456-hashed"
        stream = openai_client.responses.create(
            model=text_model_id,
            input=[{"role": "user", "content": "What is the capital of Italy?"}],
            safety_identifier=safety_id,
            stream=True,
        )

        chunks = list(stream)
        validator = StreamingValidator(chunks)
        validator.assert_basic_event_sequence()
        validator.validate_event_structure()

        # Verify safety identifier is in the created event
        created_events = [e for e in chunks if e.type == "response.created"]
        assert len(created_events) == 1
        assert created_events[0].response.safety_identifier == safety_id

        # Verify safety identifier is in the completed event
        completed_events = [e for e in chunks if e.type == "response.completed"]
        assert len(completed_events) == 1
        assert completed_events[0].response.safety_identifier == safety_id

    def test_openai_response_with_safety_identifier_and_previous_response(self, openai_client, text_model_id):
        """Test that safety_identifier works correctly with previous_response_id."""
        safety_id = "user-789-hashed"

        # Create first response
        response1 = openai_client.responses.create(
            model=text_model_id,
            input=[{"role": "user", "content": "What is 5+5?"}],
            safety_identifier=safety_id,
        )

        assert response1.id.startswith("resp_")
        assert response1.safety_identifier == safety_id

        # Create second response referencing the first one with the same safety identifier
        response2 = openai_client.responses.create(
            model=text_model_id,
            input=[{"role": "user", "content": "What is 7+7?"}],
            previous_response_id=response1.id,
            safety_identifier=safety_id,
        )

        assert response2.id.startswith("resp_")
        assert response2.safety_identifier == safety_id
        assert len(response2.output_text.strip()) > 0

    def test_openai_response_with_truncation_disabled(self, openai_client, text_model_id):
        """Test OpenAI response with truncation set to disabled."""
        response = openai_client.responses.create(
            model=text_model_id,
            input=[{"role": "user", "content": "What is the largest ocean on Earth?"}],
            truncation="disabled",
        )

        assert response.id.startswith("resp_")
        assert len(response.output_text.strip()) > 0
        assert response.truncation == "disabled"

    def test_openai_response_with_truncation_disabled_streaming(self, openai_client, text_model_id):
        """Test OpenAI response with truncation disabled in streaming mode."""
        stream = openai_client.responses.create(
            model=text_model_id,
            input=[{"role": "user", "content": "What is the smallest continent?"}],
            truncation="disabled",
            stream=True,
        )

        chunks = list(stream)
        validator = StreamingValidator(chunks)
        validator.assert_basic_event_sequence()
        validator.validate_event_structure()

        # Verify truncation is in the created event
        created_events = [e for e in chunks if e.type == "response.created"]
        assert len(created_events) == 1
        assert created_events[0].response.truncation == "disabled"

        # Verify truncation is in the completed event
        completed_events = [e for e in chunks if e.type == "response.completed"]
        assert len(completed_events) == 1
        assert completed_events[0].response.truncation == "disabled"

    def test_openai_response_with_truncation_and_previous_response(self, openai_client, text_model_id):
        """Test that truncation works correctly with previous_response_id."""
        # Create first response
        response1 = openai_client.responses.create(
            model=text_model_id,
            input=[{"role": "user", "content": "What is 4+4?"}],
            truncation="disabled",
        )

        assert response1.id.startswith("resp_")
        assert response1.truncation == "disabled"

        # Create second response referencing the first one
        response2 = openai_client.responses.create(
            model=text_model_id,
            input=[{"role": "user", "content": "What is 6+6?"}],
            previous_response_id=response1.id,
            truncation="disabled",
        )

        assert response2.id.startswith("resp_")
        assert response2.truncation == "disabled"
        assert len(response2.output_text.strip()) > 0

    def test_openai_response_with_truncation_auto_error(self, openai_client, text_model_id):
        """Test that truncation='auto' returns an error since it is not yet supported."""
        with pytest.raises(Exception) as exc_info:
            openai_client.responses.create(
                model=text_model_id,
                input=[{"role": "user", "content": "Hello"}],
                truncation="auto",
            )

        error_message = str(exc_info.value).lower()
        assert "truncation" in error_message or "auto" in error_message or "not supported" in error_message

    def test_openai_response_with_top_p(self, openai_client, text_model_id):
        """Test OpenAI response with top_p parameter."""
        response = openai_client.responses.create(
            model=text_model_id,
            input=[{"role": "user", "content": "What is the largest ocean on Earth?"}],
            top_p=0.9,
        )

        assert response.id.startswith("resp_")
        assert len(response.output_text.strip()) > 0
        assert response.top_p == 0.9

    def test_openai_response_with_top_p_streaming(self, openai_client, text_model_id):
        """Test OpenAI response with top_p in streaming mode."""
        stream = openai_client.responses.create(
            model=text_model_id,
            input=[{"role": "user", "content": "What is the smallest continent?"}],
            top_p=0.8,
            stream=True,
        )

        chunks = list(stream)
        validator = StreamingValidator(chunks)
        validator.assert_basic_event_sequence()
        validator.validate_event_structure()

        # Verify top_p is in the created event
        created_events = [e for e in chunks if e.type == "response.created"]
        assert len(created_events) == 1
        assert created_events[0].response.top_p == 0.8

        # Verify top_p is in the completed event
        completed_events = [e for e in chunks if e.type == "response.completed"]
        assert len(completed_events) == 1
        assert completed_events[0].response.top_p == 0.8

    def test_openai_response_with_top_p_and_previous_response(self, openai_client, text_model_id):
        """Test that top_p works correctly with previous_response_id."""
        # Create first response
        response1 = openai_client.responses.create(
            model=text_model_id,
            input=[{"role": "user", "content": "What is 4+4?"}],
            top_p=0.7,
        )

        assert response1.id.startswith("resp_")
        assert response1.top_p == 0.7

        # Create second response referencing the first one with the same top_p
        response2 = openai_client.responses.create(
            model=text_model_id,
            input=[{"role": "user", "content": "What is 6+6?"}],
            previous_response_id=response1.id,
            top_p=0.7,
        )

        assert response2.id.startswith("resp_")
        assert response2.top_p == 0.7
        assert len(response2.output_text.strip()) > 0

    def test_openai_response_with_top_logprobs(self, openai_client, client_with_models, text_model_id):
        """Test OpenAI response with top_logprobs parameter."""
        skip_if_provider_is_vertexai(client_with_models, text_model_id, "does not support logprobs")
        response = openai_client.responses.create(
            model=text_model_id,
            input=[{"role": "user", "content": "What is the largest ocean on Earth?"}],
            top_logprobs=3,
        )

        assert response.id.startswith("resp_")
        assert len(response.output_text.strip()) > 0
        assert response.top_logprobs == 3

    def test_openai_response_with_top_logprobs_streaming(self, openai_client, client_with_models, text_model_id):
        """Test OpenAI response with top_logprobs in streaming mode."""
        skip_if_provider_is_vertexai(client_with_models, text_model_id, "does not support logprobs")
        stream = openai_client.responses.create(
            model=text_model_id,
            input=[{"role": "user", "content": "What is the smallest continent?"}],
            top_logprobs=5,
            stream=True,
        )

        chunks = list(stream)
        validator = StreamingValidator(chunks)
        validator.assert_basic_event_sequence()
        validator.validate_event_structure()

        # Verify top_logprobs is in the created event
        created_events = [e for e in chunks if e.type == "response.created"]
        assert len(created_events) == 1
        assert created_events[0].response.top_logprobs == 5

        # Verify top_logprobs is in the completed event
        completed_events = [e for e in chunks if e.type == "response.completed"]
        assert len(completed_events) == 1
        assert completed_events[0].response.top_logprobs == 5

    def test_openai_response_with_top_logprobs_and_previous_response(
        self, openai_client, client_with_models, text_model_id
    ):
        """Test that top_logprobs works correctly with previous_response_id."""
        skip_if_provider_is_vertexai(client_with_models, text_model_id, "does not support logprobs")
        # Create first response
        response1 = openai_client.responses.create(
            model=text_model_id,
            input=[{"role": "user", "content": "What is 4+4?"}],
            top_logprobs=3,
        )

        assert response1.id.startswith("resp_")
        assert response1.top_logprobs == 3

        # Create second response referencing the first one with the same top_logprobs
        response2 = openai_client.responses.create(
            model=text_model_id,
            input=[{"role": "user", "content": "What is 6+6?"}],
            previous_response_id=response1.id,
            top_logprobs=3,
        )

        assert response2.id.startswith("resp_")
        assert response2.top_logprobs == 3
        assert len(response2.output_text.strip()) > 0

    def _function_tools(self):
        """Return a pair of function tools for parallel tool call testing."""
        return [
            {
                "type": "function",
                "name": "get_weather",
                "description": "Get weather information for a specified location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city name (e.g., 'New York', 'London')",
                        },
                    },
                },
            },
            {
                "type": "function",
                "name": "get_time",
                "description": "Get current time for a specified location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city name (e.g., 'New York', 'London')",
                        },
                    },
                },
            },
        ]

    def test_openai_response_with_parallel_tool_calls_enabled(self, openai_client, text_model_id):
        """Test that parallel_tool_calls=True produces multiple function calls."""
        if "watsonx" in text_model_id:
            pytest.skip("WatsonX does not reliably produce parallel tool calls.")

        response = openai_client.responses.create(
            model=text_model_id,
            input="What is the weather in Paris and the current time in London?",
            tools=self._function_tools(),
            parallel_tool_calls=True,
        )

        assert response.id.startswith("resp_")
        assert response.parallel_tool_calls is True

        # With parallel_tool_calls enabled, expect two function calls
        function_calls = [o for o in response.output if o.type == "function_call"]
        assert len(function_calls) == 2
        call_names = {c.name for c in function_calls}
        assert "get_weather" in call_names
        assert "get_time" in call_names

    def test_openai_response_with_parallel_tool_calls_disabled(self, openai_client, client_with_models, text_model_id):
        """Test that parallel_tool_calls=False produces only one function call."""
        if text_model_id.startswith("watsonx/"):
            pytest.skip("WatsonX does not support parallel_tool_calls parameter")
        skip_if_provider_is_vertexai(client_with_models, text_model_id, "does not respect parallel_tool_calls=False")
        response = openai_client.responses.create(
            model=text_model_id,
            input="What is the weather in Paris and the current time in London?",
            tools=self._function_tools(),
            parallel_tool_calls=False,
        )

        assert response.id.startswith("resp_")
        assert response.parallel_tool_calls is False

        # With parallel_tool_calls disabled, expect only one function call
        function_calls = [o for o in response.output if o.type == "function_call"]
        assert len(function_calls) == 1

    def test_openai_response_with_parallel_tool_calls_disabled_streaming(self, openai_client, text_model_id):
        """Test parallel_tool_calls disabled in streaming mode with function tools."""
        if text_model_id.startswith("watsonx/"):
            pytest.skip("WatsonX does not support parallel_tool_calls parameter")
        stream = openai_client.responses.create(
            model=text_model_id,
            input="What is the weather in Paris and the current time in London?",
            tools=self._function_tools(),
            parallel_tool_calls=False,
            stream=True,
        )

        chunks = list(stream)
        validator = StreamingValidator(chunks)
        validator.assert_basic_event_sequence()
        validator.validate_event_structure()

        # Verify parallel_tool_calls is in the created event
        created_events = [e for e in chunks if e.type == "response.created"]
        assert len(created_events) == 1
        assert created_events[0].response.parallel_tool_calls is False

        # Verify parallel_tool_calls is in the completed event
        completed_events = [e for e in chunks if e.type == "response.completed"]
        assert len(completed_events) == 1
        assert completed_events[0].response.parallel_tool_calls is False

    def test_openai_response_with_parallel_tool_calls_and_previous_response(self, openai_client, text_model_id):
        """Test that parallel_tool_calls works correctly with previous_response_id."""
        # Create first response without tools so the conversation can be chained
        response1 = openai_client.responses.create(
            model=text_model_id,
            input=[{"role": "user", "content": "What is 4+4?"}],
            parallel_tool_calls=False,
        )

        assert response1.id.startswith("resp_")
        assert response1.parallel_tool_calls is False

        # Create second response referencing the first one with the same parallel_tool_calls
        response2 = openai_client.responses.create(
            model=text_model_id,
            input=[{"role": "user", "content": "What is 6+6?"}],
            previous_response_id=response1.id,
            parallel_tool_calls=False,
        )

        assert response2.id.startswith("resp_")
        assert response2.parallel_tool_calls is False

    def test_openai_response_background_returns_queued(self, openai_client, text_model_id):
        """Test that background=True returns immediately with queued status."""
        response = openai_client.responses.create(
            model=text_model_id,
            input="What is 2+2?",
            background=True,
        )

        # Should return immediately with queued status
        assert response.status == "queued"
        assert response.background is True
        assert response.id.startswith("resp_")
        # Output should be empty initially
        assert len(response.output) == 0

    def test_openai_response_background_completes(self, openai_client, text_model_id):
        """Test that a background response eventually completes."""
        if text_model_id.startswith("watsonx/"):
            pytest.skip("WatsonX rate limits cause background responses to fail")
        response = openai_client.responses.create(
            model=text_model_id,
            input="Say hello",
            background=True,
        )

        assert response.status == "queued"
        response_id = response.id

        # Poll for completion (max 60 seconds)
        max_wait = 60
        poll_interval = 1
        elapsed = 0

        while elapsed < max_wait:
            time.sleep(poll_interval)
            elapsed += poll_interval

            retrieved = openai_client.responses.retrieve(response_id=response_id)

            if retrieved.status == "completed":
                assert retrieved.background is True
                assert len(retrieved.output) > 0
                assert len(retrieved.output_text) > 0
                return

            if retrieved.status == "failed":
                pytest.fail(f"Background response failed: {retrieved.error}")

            # Status should be queued or in_progress while processing
            assert retrieved.status in ("queued", "in_progress")

        pytest.fail(f"Background response did not complete within {max_wait} seconds")

    def test_openai_response_background_and_stream_mutually_exclusive(self, openai_client, text_model_id):
        """Test that background=True and stream=True cannot be used together."""
        with pytest.raises(Exception) as exc_info:
            openai_client.responses.create(
                model=text_model_id,
                input="Hello",
                background=True,
                stream=True,
            )

        error_msg = str(exc_info.value).lower()
        assert "background" in error_msg or "stream" in error_msg

    def test_openai_response_background_false_is_synchronous(self, openai_client, text_model_id):
        """Test that background=False returns a completed response synchronously."""
        response = openai_client.responses.create(
            model=text_model_id,
            input="What is 1+1?",
            background=False,
        )

        assert response.status == "completed"
        assert response.background is False
        assert len(response.output) > 0

    def test_cancel_queued_or_in_progress_response(self, openai_client, text_model_id):
        """Test cancelling a background response that is queued or in progress."""
        if text_model_id.startswith("watsonx/"):
            pytest.skip("WatsonX rate limits cause cancel tests to fail")
        # Create a background response
        response = openai_client.responses.create(
            model=text_model_id,
            input="Write a detailed 5000 word essay about quantum physics and the nature of reality.",
            background=True,
        )

        assert response.status == "queued"
        response_id = response.id

        # Cancel immediately - in replay mode, background worker starts very quickly
        openai_client.responses.cancel(response_id=response_id)

        # Poll for cancelled status (background worker may have picked up task)
        max_wait = 5
        poll_interval = 0.1
        elapsed = 0

        while elapsed < max_wait:
            time.sleep(poll_interval)
            elapsed += poll_interval

            retrieved = openai_client.responses.retrieve(response_id=response_id)
            if retrieved.status == "cancelled":
                return

            # In replay mode, worker may have started processing before cancel completed
            assert retrieved.status in ("queued", "in_progress", "cancelled"), (
                f"Unexpected status '{retrieved.status}' - expected queued/in_progress/cancelled"
            )

        pytest.fail(f"Response did not transition to cancelled within {max_wait} seconds")

    def test_cancel_already_cancelled_is_idempotent(self, openai_client, text_model_id):
        """Test that cancelling an already-cancelled response is idempotent."""
        if text_model_id.startswith("watsonx/"):
            pytest.skip("WatsonX rate limits cause cancel tests to fail")
        # Create and cancel a background response
        response = openai_client.responses.create(
            model=text_model_id,
            input="Write a long story.",
            background=True,
        )

        response_id = response.id
        openai_client.responses.cancel(response_id=response_id)

        # Poll for cancelled status
        max_wait = 5
        poll_interval = 0.1
        elapsed = 0

        while elapsed < max_wait:
            time.sleep(poll_interval)
            elapsed += poll_interval

            retrieved = openai_client.responses.retrieve(response_id=response_id)
            if retrieved.status == "cancelled":
                break

            assert retrieved.status in ("queued", "in_progress", "cancelled"), f"Unexpected status '{retrieved.status}'"
        else:
            pytest.fail(f"Response did not transition to cancelled within {max_wait} seconds")

        # Cancel again - should return same state without error
        cancelled_again = openai_client.responses.cancel(response_id=response_id)
        assert cancelled_again.id == response_id
        assert cancelled_again.status == "cancelled"

    def test_cancel_completed_response_fails(self, openai_client, client_with_models, text_model_id):
        """Test that cancelling a completed response returns 409 Conflict."""
        if text_model_id.startswith("watsonx/"):
            pytest.skip("WatsonX rate limits cause cancel tests to fail")
        skip_if_provider_is_vertexai(
            client_with_models, text_model_id, "returns 500 instead of 409 for cancel on completed response"
        )
        # Create a synchronous (completed) response
        response = openai_client.responses.create(
            model=text_model_id,
            input="Say hello",
            background=False,
        )

        assert response.status == "completed"
        response_id = response.id

        # Try to cancel it - should fail with 409
        with pytest.raises(Exception) as exc_info:
            openai_client.responses.cancel(response_id=response_id)

        # Check for conflict error (different clients may raise different exceptions)
        error_str = str(exc_info.value).lower()
        assert "409" in error_str or "conflict" in error_str or "cannot cancel" in error_str

    def test_cancel_nonexistent_response_fails(self, openai_client, text_model_id):
        """Test that cancelling a non-existent response returns 404."""
        if text_model_id.startswith("watsonx/"):
            pytest.skip("WatsonX rate limits cause cancel tests to fail")
        fake_id = "resp_fake_nonexistent_id"

        with pytest.raises(Exception) as exc_info:
            openai_client.responses.cancel(response_id=fake_id)

        # Check for not found error
        error_str = str(exc_info.value).lower()
        assert "404" in error_str or "not found" in error_str

    def test_cancel_prevents_completion(self, openai_client, text_model_id):
        """Test that a cancelled response does not complete."""
        if text_model_id.startswith("watsonx/"):
            pytest.skip("WatsonX rate limits cause cancel tests to fail")
        # Create a background response
        response = openai_client.responses.create(
            model=text_model_id,
            input="Write a detailed essay.",
            background=True,
        )

        response_id = response.id
        assert response.status == "queued"

        # Cancel immediately
        cancelled = openai_client.responses.cancel(response_id=response_id)
        assert cancelled.status == "cancelled"

        # Poll to verify it stays cancelled and doesn't complete
        max_wait = 5
        poll_interval = 0.5
        elapsed = 0

        while elapsed < max_wait:
            time.sleep(poll_interval)
            elapsed += poll_interval

            retrieved = openai_client.responses.retrieve(response_id=response_id)
            assert retrieved.status == "cancelled", f"Expected 'cancelled' but got '{retrieved.status}'"
            assert len(retrieved.output) == 0

    def _skip_service_tier_for_unsupported(self, client_with_models, text_model_id):
        if text_model_id.startswith("azure/"):
            pytest.skip("Azure OpenAI does not support the service_tier parameter")
        if text_model_id.startswith("watsonx/"):
            pytest.skip("WatsonX does not support the service_tier parameter")
        skip_if_provider_is_vertexai(client_with_models, text_model_id, "does not support the service_tier parameter")
        if text_model_id.startswith("vllm/"):
            pytest.skip("vLLM does not support the service_tier parameter")

    def test_openai_response_with_service_tier_auto(self, openai_client, client_with_models, text_model_id):
        """Test OpenAI response with service_tier='auto'.

        When 'auto' is requested, the provider decides the actual tier (e.g. default, priority),
        so we only assert the response has a non-null service_tier.
        """
        self._skip_service_tier_for_unsupported(client_with_models, text_model_id)

        response = openai_client.responses.create(
            model=text_model_id,
            input=[{"role": "user", "content": "What is the speed of light?"}],
            service_tier="auto",
        )

        assert response.id.startswith("resp_")
        assert len(response.output_text.strip()) > 0
        assert response.service_tier is not None

    @pytest.mark.parametrize("service_tier", ["default", "priority"])
    def test_openai_response_with_service_tier(self, openai_client, client_with_models, text_model_id, service_tier):
        """Test OpenAI response with explicit service_tier values that should be preserved."""
        self._skip_service_tier_for_unsupported(client_with_models, text_model_id)

        response = openai_client.responses.create(
            model=text_model_id,
            input=[{"role": "user", "content": "What is the speed of light?"}],
            service_tier=service_tier,
        )

        assert response.id.startswith("resp_")
        assert len(response.output_text.strip()) > 0
        assert response.service_tier == service_tier

    def test_openai_response_with_service_tier_flex(self, openai_client, client_with_models, text_model_id):
        """Test OpenAI response with service_tier='flex'.

        The flex tier may not be supported by all providers (e.g. OpenAI rejects it
        for certain models). This test verifies the request is accepted with the
        exact tier preserved, or properly rejected.
        """
        self._skip_service_tier_for_unsupported(client_with_models, text_model_id)

        try:
            response = openai_client.responses.create(
                model=text_model_id,
                input=[{"role": "user", "content": "What is the speed of light?"}],
                service_tier="flex",
            )
            assert response.id.startswith("resp_")
            assert response.service_tier == "flex"
        except Exception as e:
            error_message = str(e).lower()
            assert "service_tier" in error_message or "invalid" in error_message

    def test_openai_response_with_service_tier_auto_streaming(self, openai_client, client_with_models, text_model_id):
        """Test OpenAI response with service_tier='auto' in streaming mode."""
        self._skip_service_tier_for_unsupported(client_with_models, text_model_id)

        stream = openai_client.responses.create(
            model=text_model_id,
            input=[{"role": "user", "content": "What is the speed of sound?"}],
            service_tier="auto",
            stream=True,
        )

        chunks = list(stream)
        validator = StreamingValidator(chunks)
        validator.assert_basic_event_sequence()
        validator.validate_event_structure()

        # Verify service_tier is in the created event
        created_events = [e for e in chunks if e.type == "response.created"]
        assert len(created_events) == 1
        assert created_events[0].response.service_tier is not None

        # Verify service_tier is in the completed event
        completed_events = [e for e in chunks if e.type == "response.completed"]
        assert len(completed_events) == 1
        assert completed_events[0].response.service_tier is not None

    @pytest.mark.parametrize("service_tier", ["default", "priority"])
    def test_openai_response_with_service_tier_streaming(
        self, openai_client, client_with_models, text_model_id, service_tier
    ):
        """Test OpenAI response with explicit service_tier values in streaming mode."""
        self._skip_service_tier_for_unsupported(client_with_models, text_model_id)

        stream = openai_client.responses.create(
            model=text_model_id,
            input=[{"role": "user", "content": "What is the speed of sound?"}],
            service_tier=service_tier,
            stream=True,
        )

        chunks = list(stream)
        validator = StreamingValidator(chunks)
        validator.assert_basic_event_sequence()
        validator.validate_event_structure()

        # Verify service_tier is preserved in the created event
        created_events = [e for e in chunks if e.type == "response.created"]
        assert len(created_events) == 1
        assert created_events[0].response.service_tier == service_tier

        # Verify service_tier is preserved in the completed event
        completed_events = [e for e in chunks if e.type == "response.completed"]
        assert len(completed_events) == 1
        assert completed_events[0].response.service_tier == service_tier

    def test_openai_response_with_service_tier_flex_streaming(self, openai_client, client_with_models, text_model_id):
        """Test OpenAI response with service_tier='flex' in streaming mode.

        The flex tier may not be supported by all providers. This test verifies
        the request is accepted with the exact tier preserved, or produces a proper failure event.
        """
        self._skip_service_tier_for_unsupported(client_with_models, text_model_id)

        stream = openai_client.responses.create(
            model=text_model_id,
            input=[{"role": "user", "content": "What is the speed of sound?"}],
            service_tier="flex",
            stream=True,
        )

        chunks = list(stream)
        validator = StreamingValidator(chunks)
        validator.assert_basic_event_sequence()
        validator.validate_event_structure()

        # The response should either complete or fail gracefully
        completed_events = [e for e in chunks if e.type == "response.completed"]
        failed_events = [e for e in chunks if e.type == "response.failed"]
        assert len(completed_events) + len(failed_events) == 1

        if completed_events:
            assert completed_events[0].response.service_tier == "flex"

    def test_openai_response_with_service_tier_auto_and_previous_response(
        self, openai_client, client_with_models, text_model_id
    ):
        """Test that service_tier='auto' works correctly with previous_response_id."""
        self._skip_service_tier_for_unsupported(client_with_models, text_model_id)

        response1 = openai_client.responses.create(
            model=text_model_id,
            input=[{"role": "user", "content": "What is 8+8?"}],
            service_tier="auto",
        )

        assert response1.id.startswith("resp_")
        assert response1.service_tier is not None

        response2 = openai_client.responses.create(
            model=text_model_id,
            input=[{"role": "user", "content": "What is 9+9?"}],
            previous_response_id=response1.id,
            service_tier="auto",
        )

        assert response2.id.startswith("resp_")
        assert response2.service_tier is not None
        assert len(response2.output_text.strip()) > 0

    @pytest.mark.parametrize("service_tier", ["default", "priority"])
    def test_openai_response_with_service_tier_and_previous_response(
        self, openai_client, client_with_models, text_model_id, service_tier
    ):
        """Test that explicit service_tier values are preserved with previous_response_id."""
        self._skip_service_tier_for_unsupported(client_with_models, text_model_id)

        response1 = openai_client.responses.create(
            model=text_model_id,
            input=[{"role": "user", "content": "What is 8+8?"}],
            service_tier=service_tier,
        )

        assert response1.id.startswith("resp_")
        assert response1.service_tier == service_tier

        response2 = openai_client.responses.create(
            model=text_model_id,
            input=[{"role": "user", "content": "What is 9+9?"}],
            previous_response_id=response1.id,
            service_tier=service_tier,
        )

        assert response2.id.startswith("resp_")
        assert response2.service_tier == service_tier
        assert len(response2.output_text.strip()) > 0

    def test_openai_response_streaming_includes_usage(self, openai_client, text_model_id):
        """Test that streaming response includes usage information.

        OGX always sets include_usage=True in the underlying chat completion
        stream_options, so usage should always be present in the completed response.
        """
        stream = openai_client.responses.create(
            model=text_model_id,
            input=[{"role": "user", "content": "What is the capital of France?"}],
            stream=True,
        )

        chunks = list(stream)
        validator = StreamingValidator(chunks)
        validator.assert_basic_event_sequence()
        validator.validate_event_structure()

        completed_events = [e for e in chunks if e.type == "response.completed"]
        assert len(completed_events) == 1

        response = completed_events[0].response
        assert len(response.output_text.strip()) > 0
        assert response.usage is not None
        assert response.usage.output_tokens > 0
        assert response.usage.total_tokens > 0

    def test_openai_response_with_stream_options_includes_usage(self, openai_client, text_model_id):
        """Test that stream_options parameter is accepted and usage is still included."""
        stream = openai_client.responses.create(
            model=text_model_id,
            input=[{"role": "user", "content": "What is the capital of Germany?"}],
            stream=True,
            stream_options={"include_obfuscation": True},
        )

        chunks = list(stream)
        validator = StreamingValidator(chunks)
        validator.assert_basic_event_sequence()
        validator.validate_event_structure()

        completed_events = [e for e in chunks if e.type == "response.completed"]
        assert len(completed_events) == 1

        response = completed_events[0].response
        assert len(response.output_text.strip()) > 0
        assert response.usage is not None
        assert response.usage.output_tokens > 0
        assert response.usage.total_tokens > 0

    def test_openai_response_with_stream_options_non_streaming(self, openai_client, text_model_id):
        """Test that stream_options is accepted in non-streaming mode."""
        response = openai_client.responses.create(
            model=text_model_id,
            input=[{"role": "user", "content": "What is the capital of Italy?"}],
            stream_options={"include_obfuscation": True},
        )

        assert response.id.startswith("resp_")
        assert len(response.output_text.strip()) > 0
        assert response.status == "completed"
        assert response.usage is not None
        assert response.usage.output_tokens > 0
        assert response.usage.total_tokens > 0

    def test_openai_response_with_stream_options_and_previous_response(self, openai_client, text_model_id):
        """Test that stream_options works correctly with previous_response_id in streaming mode."""
        response1 = openai_client.responses.create(
            model=text_model_id,
            input=[{"role": "user", "content": "What is 3+3?"}],
        )

        assert response1.id.startswith("resp_")

        stream = openai_client.responses.create(
            model=text_model_id,
            input=[{"role": "user", "content": "What is 5+5?"}],
            previous_response_id=response1.id,
            stream=True,
            stream_options={"include_obfuscation": True},
        )

        chunks = list(stream)
        validator = StreamingValidator(chunks)
        validator.assert_basic_event_sequence()
        validator.validate_event_structure()

        completed_events = [e for e in chunks if e.type == "response.completed"]
        assert len(completed_events) == 1

        response = completed_events[0].response
        assert len(response.output_text.strip()) > 0
        assert response.usage is not None
        assert response.usage.output_tokens > 0
        assert response.usage.total_tokens > 0

    def test_openai_response_incomplete_details_null_when_completed(self, openai_client, text_model_id):
        """Test that a completed response has incomplete_details as None."""
        response = openai_client.responses.create(
            model=text_model_id,
            input=[{"role": "user", "content": "What is 2+2?"}],
        )

        assert response.id.startswith("resp_")
        assert response.status == "completed"
        assert response.incomplete_details is None

    def test_openai_response_incomplete_details_length(self, openai_client, client_with_models, text_model_id):
        """Test incomplete_details.reason is 'length' when chat completion returns finish_reason='length'.

        A small max_output_tokens with a long prompt causes the provider to truncate
        the output in a single inference call, returning finish_reason='length'.
        """
        skip_if_provider_is_vertexai(
            client_with_models,
            text_model_id,
            "does not reliably return finish_reason='length' with small max_output_tokens",
        )
        response = openai_client.responses.create(
            model=text_model_id,
            input=[
                {
                    "role": "user",
                    "content": "Write a very long and detailed essay about the entire history of the Roman Empire from founding to fall.",
                }
            ],
            max_output_tokens=16,
        )

        assert response.id.startswith("resp_")
        assert response.status == "incomplete"
        assert response.incomplete_details is not None
        assert response.incomplete_details.reason == "length"

    def test_openai_response_incomplete_details_length_streaming(
        self, openai_client, client_with_models, text_model_id
    ):
        """Test streaming incomplete_details.reason is 'length' when chat completion returns finish_reason='length'."""
        skip_if_provider_is_vertexai(
            client_with_models,
            text_model_id,
            "does not reliably return finish_reason='length' with small max_output_tokens",
        )
        stream = openai_client.responses.create(
            model=text_model_id,
            input=[
                {
                    "role": "user",
                    "content": "Write a very long and detailed essay about the entire history of the Roman Empire from founding to fall.",
                }
            ],
            max_output_tokens=16,
            stream=True,
        )

        chunks = list(stream)
        validator = StreamingValidator(chunks)
        validator.assert_basic_event_sequence()

        incomplete_events = [e for e in chunks if e.type == "response.incomplete"]
        assert len(incomplete_events) == 1
        assert incomplete_events[0].response.status == "incomplete"
        assert incomplete_events[0].response.incomplete_details is not None
        assert incomplete_events[0].response.incomplete_details.reason == "length"

    def test_openai_response_incomplete_details_max_iterations_exceeded(self, openai_client, text_model_id):
        """Test incomplete_details.reason is 'max_iterations_exceeded' when the agent loop
        hits the max_infer_iters limit.

        This uses web_search (a server-side tool) with max_infer_iters=1 so the loop
        exits after the first tool-calling iteration.
        Note: _function_tools cannot be used here because function (client-side) tools
        break the loop immediately, so n_iter never increments.
        """
        response = openai_client.responses.create(
            model=text_model_id,
            input="Search for the latest news about artificial intelligence.",
            tools=[{"type": "web_search"}],
            extra_body={"max_infer_iters": 1},
        )

        assert response.id.startswith("resp_")
        assert response.status == "incomplete"
        assert response.incomplete_details is not None
        assert response.incomplete_details.reason == "max_iterations_exceeded"

    def test_openai_response_incomplete_details_max_iterations_exceeded_streaming(self, openai_client, text_model_id):
        """Test streaming incomplete_details.reason is 'max_iterations_exceeded' when the agent loop
        hits the max_infer_iters limit."""
        stream = openai_client.responses.create(
            model=text_model_id,
            input="Search for the latest news about artificial intelligence.",
            tools=[{"type": "web_search"}],
            extra_body={"max_infer_iters": 1},
            stream=True,
        )

        chunks = list(stream)
        validator = StreamingValidator(chunks)
        validator.assert_basic_event_sequence()

        incomplete_events = [e for e in chunks if e.type == "response.incomplete"]
        assert len(incomplete_events) == 1
        assert incomplete_events[0].response.status == "incomplete"
        assert incomplete_events[0].response.incomplete_details is not None
        assert incomplete_events[0].response.incomplete_details.reason == "max_iterations_exceeded"

    @staticmethod
    def _is_reasoning_model(model_id: str) -> bool:
        """Check if the model supports reasoning_effort based on model name patterns."""
        # Strip provider prefix (e.g., "openai/", "azure/") to get base model name
        base_model = model_id.split("/")[-1] if "/" in model_id else model_id
        # OpenAI reasoning models: o1, o3, o4, etc.
        reasoning_prefixes = ("o1", "o3", "o4")
        return base_model.startswith(reasoning_prefixes)

    def _skip_reasoning_effort_for_unsupported(self, text_model_id):
        if not self._is_reasoning_model(text_model_id):
            pytest.skip(f"Model {text_model_id} does not support the reasoning_effort parameter")

    @pytest.mark.parametrize("effort", ["low", "medium", "high"])
    def test_openai_response_reasoning_effort(self, openai_client, text_model_id, effort):
        """Test that reasoning.effort is accepted and reflected in the response."""
        self._skip_reasoning_effort_for_unsupported(text_model_id)
        response = openai_client.responses.create(
            model=text_model_id,
            input=[{"role": "user", "content": "What is 2+2?"}],
            reasoning={"effort": effort},
        )

        assert response.id.startswith("resp_")
        assert response.status == "completed"
        assert len(response.output_text.strip()) > 0
        assert response.reasoning is not None
        assert response.reasoning.effort == effort

    @pytest.mark.parametrize("effort", ["low", "medium", "high"])
    def test_openai_response_reasoning_effort_streaming(self, openai_client, text_model_id, effort):
        """Test that reasoning.effort works correctly in streaming mode."""
        self._skip_reasoning_effort_for_unsupported(text_model_id)
        stream = openai_client.responses.create(
            model=text_model_id,
            input=[{"role": "user", "content": "What is 2+2?"}],
            reasoning={"effort": effort},
            stream=True,
        )

        chunks = list(stream)
        validator = StreamingValidator(chunks)
        validator.assert_basic_event_sequence()

        completed_events = [e for e in chunks if e.type == "response.completed"]
        assert len(completed_events) == 1

        response = completed_events[0].response
        assert response.status == "completed"
        assert len(response.output_text.strip()) > 0
        assert response.reasoning is not None
        assert response.reasoning.effort == effort
        assert response.usage is not None
        assert response.usage.output_tokens > 0
        assert response.usage.total_tokens > 0
