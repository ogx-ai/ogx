# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import time

import pytest

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

    def test_openai_response_with_small_max_output_tokens(self, openai_client, text_model_id):
        """Test response with very small max_output_tokens to trigger potential truncation."""
        if text_model_id.startswith("watsonx/"):
            pytest.skip("WatsonX does not support max_output_tokens parameter")
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

    def test_openai_response_with_parallel_tool_calls_disabled(self, openai_client, text_model_id):
        """Test that parallel_tool_calls=False produces only one function call."""
        if text_model_id.startswith("watsonx/"):
            pytest.skip("WatsonX does not support parallel_tool_calls parameter")
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
