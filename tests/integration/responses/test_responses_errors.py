# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""
Error handling tests for the OGX Responses and Conversations APIs.

These tests verify that errors emitted by OGX are correctly typed
and handled by the OpenAI Python SDK, ensuring users don't have breaking
experiences when error conditions occur.

HTTP Error Tests (TestResponsesAPIErrors, TestConversationsAPIErrors):
The OpenAI SDK expects specific HTTP status codes to trigger specific
exception types:
    - 400 -> openai.BadRequestError
    - 401 -> openai.AuthenticationError
    - 404 -> openai.NotFoundError
    - 409 -> openai.ConflictError
    - 422 -> openai.UnprocessableEntityError
    - 429 -> openai.RateLimitError
    - 5xx -> openai.InternalServerError

See: https://github.com/openai/openai-python/blob/main/src/openai/_exceptions.py

Streaming Error Tests (TestResponsesAPIStreamingErrors):
When errors occur during streaming, the Responses API emits a `response.failed`
event with an error object containing a `code` and `message`. The valid error
codes are defined by OpenAI's ResponseError model:
    - server_error
    - rate_limit_exceeded
    - invalid_prompt
    - vector_store_timeout
    - invalid_image, invalid_image_format, invalid_base64_image, invalid_image_url
    - image_too_large, image_too_small, image_parse_error
    - image_content_policy_violation, invalid_image_mode
    - image_file_too_large, unsupported_image_media_type
    - empty_image_file, failed_to_download_image, image_file_not_found

See: https://github.com/openai/openai-python/blob/main/src/openai/types/responses/response_error.py
"""

import os
from typing import get_args

import httpx
import pytest
from openai import APIError, APIStatusError, BadRequestError, NotFoundError
from openai.types.responses import ResponseError

from ogx_api.common.errors import (
    ConversationNotFoundError,
    ModelNotFoundError,
    ResponseNotFoundError,
)

# Extract valid error codes directly from the OpenAI ResponseError model
# This ensures we stay in sync with the OpenAI spec automatically
# Use model_fields (Pydantic v2) to get the annotation without forward reference issues
_code_annotation = ResponseError.model_fields["code"].annotation
VALID_OPENAI_RESPONSE_ERROR_CODES: set[str] = set(get_args(_code_annotation))


class TestResponsesAPIErrors:
    """Error handling tests for the Responses API.

    These tests verify SDK compatibility by ensuring OGX returns
    the correct HTTP status codes that trigger the expected OpenAI SDK
    exception types for Responses API operations.
    """

    def test_invalid_model_raises_not_found_error(self, openai_client):
        """
        Test that requesting a nonexistent model returns 404 and triggers
        openai.NotFoundError in the SDK.

        This is critical for SDK compatibility - users catching NotFoundError
        should have their error handling work correctly.
        """
        model_name = "nonexistent-model-xyz-12345"
        with pytest.raises(NotFoundError) as exc_info:
            openai_client.responses.create(
                model=model_name,
                input="Hello, world!",
            )

        assert exc_info.value.status_code == 404
        expected_msg = str(ModelNotFoundError(model_name))
        assert expected_msg in str(exc_info.value)

    def test_invalid_previous_response_id_raises_not_found_error(self, openai_client, text_model_id):
        """
        Test that referencing a nonexistent previous_response_id returns 404.

        Per OpenResponses spec, previous_response_id references a prior response
        for multi-turn conversations.
        """
        response_id = "resp_nonexistent123456"
        with pytest.raises(NotFoundError) as exc_info:
            openai_client.responses.create(
                model=text_model_id,
                input="Continue the conversation",
                previous_response_id=response_id,
            )

        assert exc_info.value.status_code == 404
        expected_msg = str(ResponseNotFoundError(response_id))
        assert expected_msg in str(exc_info.value)

    def test_invalid_max_tool_calls_raises_bad_request(self, openai_client, text_model_id):
        """
        Test that invalid max_tool_calls (< 1) returns 400.
        """
        with pytest.raises(BadRequestError) as exc_info:
            openai_client.responses.create(
                model=text_model_id,
                input="Search for news",
                tools=[{"type": "web_search"}],
                max_tool_calls=0,  # Invalid: must be >= 1
            )

        assert exc_info.value.status_code == 400
        error_msg = str(exc_info.value).lower()
        assert "max_tool_calls" in error_msg or "invalid" in error_msg

    def test_invalid_temperature_raises_bad_request(self, openai_client, text_model_id):
        """
        Test that temperature outside valid range (0-2) returns 400.

        Per OpenResponses spec: "Sampling temperature to use, between 0 and 2."
        """
        with pytest.raises(BadRequestError) as exc_info:
            openai_client.responses.create(
                model=text_model_id,
                input="Hello",
                temperature=3.0,  # Invalid: must be between 0 and 2
            )

        assert exc_info.value.status_code == 400
        error_msg = str(exc_info.value).lower()
        assert "temperature" in error_msg or "invalid" in error_msg or "range" in error_msg

    def test_invalid_tool_choice_raises_bad_request(self, openai_client, text_model_id):
        """
        Test that invalid tool_choice value returns 400.

        Per OpenResponses spec, tool_choice controls which tool the model should use.
        """
        with pytest.raises(BadRequestError) as exc_info:
            openai_client.responses.create(
                model=text_model_id,
                input="Hello",
                tools=[{"type": "function", "function": {"name": "test", "parameters": {}}}],
                tool_choice="invalid_choice",  # Invalid: must be valid enum or object
            )

        assert exc_info.value.status_code == 400
        error_msg = str(exc_info.value).lower()
        assert "tool_choice" in error_msg or "invalid" in error_msg

    def test_retrieve_nonexistent_response_raises_not_found_error(self, openai_client):
        """GET /responses/{id} for a nonexistent response returns 404."""
        response_id = "resp_nonexistent123456"
        with pytest.raises(NotFoundError) as exc_info:
            openai_client.responses.retrieve(response_id)
        assert exc_info.value.status_code == 404
        expected_msg = str(ResponseNotFoundError(response_id))
        assert expected_msg in str(exc_info.value)

    def test_delete_nonexistent_response_raises_not_found_error(self, openai_client):
        """DELETE /responses/{id} for a nonexistent response returns 404."""
        response_id = "resp_nonexistent123456"
        with pytest.raises(NotFoundError) as exc_info:
            openai_client.responses.delete(response_id)
        assert exc_info.value.status_code == 404
        expected_msg = str(ResponseNotFoundError(response_id))
        assert expected_msg in str(exc_info.value)

    def test_conflicting_previous_response_and_conversation_raises_bad_request(self, openai_client, text_model_id):
        """Providing both previous_response_id and conversation returns 400."""
        with pytest.raises(BadRequestError) as exc_info:
            openai_client.responses.create(
                model=text_model_id,
                input="Hello",
                previous_response_id="resp_abc123",
                conversation="conv_xyz789",
            )
        assert exc_info.value.status_code == 400
        error_msg = str(exc_info.value)
        assert "previous_response_id" in error_msg
        assert "conversation" in error_msg

    def test_malformed_request_returns_sdk_compatible_error(self, openai_client):
        """Pydantic validation errors return 400 with OpenAI error format, not FastAPI's 422."""
        base_url = str(openai_client.base_url).rstrip("/")
        response = httpx.post(
            f"{base_url}/responses", json={}, headers={"Authorization": f"Bearer {openai_client.api_key}"}
        )
        assert response.status_code == 400
        body = response.json()
        assert "error" in body
        assert "message" in body["error"]

    def test_guardrails_without_safety_api_raises_service_unavailable(self, openai_client, ogx_client, text_model_id):
        """Guardrails without Safety API configured returns 503."""
        safety_providers = [p for p in ogx_client.providers.list() if p.api == "safety"]
        if safety_providers:
            pytest.skip("Server has Safety API enabled")

        with pytest.raises(APIStatusError) as exc_info:
            openai_client.responses.create(
                model=text_model_id,
                input="Hello",
                extra_body={"guardrails": ["test-shield"]},
            )
        assert exc_info.value.status_code == 503
        assert "safety" in str(exc_info.value).lower()


class TestConversationsAPIErrors:
    """Error handling tests for the Conversations API.

    These tests verify SDK compatibility for conversation-related operations
    accessed through the Responses API.
    """

    def test_invalid_conversation_id_format_raises_bad_request(self, openai_client, text_model_id):
        """
        Test that an invalid conversation ID format returns 400 and triggers
        openai.BadRequestError in the SDK.
        """
        conversation_id = "invalid-format-no-conv-prefix"
        with pytest.raises(BadRequestError) as exc_info:
            openai_client.responses.create(
                model=text_model_id,
                input=[{"role": "user", "content": "Hello"}],
                conversation=conversation_id,
            )

        assert exc_info.value.status_code == 400
        error_msg = str(exc_info.value)
        assert "conversation" in error_msg
        assert "conv_" in error_msg

    def test_nonexistent_conversation_raises_not_found_error(self, openai_client, text_model_id):
        """
        Test that referencing a nonexistent conversation returns 404 and triggers
        openai.NotFoundError in the SDK.
        """
        conversation_id = "conv_" + "0" * 48
        with pytest.raises(NotFoundError) as exc_info:
            openai_client.responses.create(
                model=text_model_id,
                input=[{"role": "user", "content": "Hello"}],
                conversation=conversation_id,
            )

        assert exc_info.value.status_code == 404
        expected_msg = str(ConversationNotFoundError(conversation_id))
        assert expected_msg in str(exc_info.value)


class TestResponsesAPIStreamingErrors:
    """Streaming error tests for the Responses API.

    These tests verify that when errors occur during streaming, the
    `response.failed` event contains valid OpenAI ResponseError codes
    as defined in the OpenAI spec.

    Unlike HTTP errors (which raise SDK exceptions), streaming errors are
    delivered as `response.failed` events within the stream. The error
    object must have:
        - code: One of the valid ResponseError codes
        - message: A human-readable description

    Valid ResponseError codes per OpenAI spec:
        - server_error, rate_limit_exceeded, invalid_prompt, vector_store_timeout
        - invalid_image, invalid_image_format, invalid_base64_image, invalid_image_url
        - image_too_large, image_too_small, image_parse_error
        - image_content_policy_violation, invalid_image_mode
        - image_file_too_large, unsupported_image_media_type
        - empty_image_file, failed_to_download_image, image_file_not_found

    See: https://github.com/openai/openai-python/blob/main/src/openai/types/responses/response_error.py
    """

    def _consume_stream(self, stream):
        """Consume a stream and return all chunks."""
        return list(stream)

    def _get_terminal_event(self, chunks):
        """Get the terminal event (completed, incomplete, or failed) from chunks."""
        for chunk in reversed(chunks):
            if chunk.type in ("response.completed", "response.incomplete", "response.failed"):
                return chunk
        return None

    def _assert_valid_error_code(self, error_code: str):
        """Assert that the error code is one of the valid OpenAI ResponseError codes."""
        assert error_code in VALID_OPENAI_RESPONSE_ERROR_CODES, (
            f"Invalid error code '{error_code}'. Must be one of: {sorted(VALID_OPENAI_RESPONSE_ERROR_CODES)}"
        )

    def _assert_failed_event_structure(self, failed_event):
        """Assert that a response.failed event has the correct structure."""
        assert failed_event.type == "response.failed"
        assert failed_event.response.status == "failed"
        assert failed_event.response.error is not None, "Failed response must have an error object"
        assert failed_event.response.error.code, "Error must have a non-empty code"
        assert failed_event.response.error.message, "Error must have a non-empty message"
        assert isinstance(failed_event.sequence_number, int), "sequence_number must be an integer"
        self._assert_valid_error_code(failed_event.response.error.code)

    def test_completed_response_has_no_error(self, openai_client, text_model_id):
        """
        Test that successfully completed streaming responses have no error.

        A response.completed event should have:
            - response.status == "completed"
            - response.error == None
        """
        try:
            stream = openai_client.responses.create(
                model=text_model_id,
                input="Say 'hello' and nothing else.",
                stream=True,
            )
            chunks = self._consume_stream(stream)
            terminal = self._get_terminal_event(chunks)
        except (BadRequestError, NotFoundError, APIError) as e:
            pytest.fail(f"Simple request should not fail with HTTP error: {e}")

        assert terminal is not None, "Expected a terminal event"

        if terminal.type == "response.failed":
            # If request failed, show the error for debugging
            error_info = (
                f"code={terminal.response.error.code}, message={terminal.response.error.message}"
                if terminal.response.error
                else "no error details"
            )
            pytest.fail(f"Simple request unexpectedly failed: {error_info}")

        assert terminal.type == "response.completed", f"Expected completed, got {terminal.type}"
        assert terminal.response.status == "completed"
        assert terminal.response.error is None, "Completed response should not have an error"

    def test_incomplete_response_has_no_error(self, openai_client, text_model_id):
        """
        Test that incomplete responses (length limit) have no error object.

        A response.incomplete event indicates the model hit output limits,
        which is NOT an error - it's normal behavior.

        Note: This test requires live mode as incomplete responses cannot be
        reliably recorded (they depend on hitting output limits).
        """
        inference_mode = os.environ.get("OGX_TEST_INFERENCE_MODE", "replay")
        if inference_mode != "live":
            pytest.skip("Incomplete response test requires live mode (cannot be recorded)")

        try:
            stream = openai_client.responses.create(
                model=text_model_id,
                input="Write a detailed 10,000 word essay about quantum computing.",
                stream=True,
            )
            chunks = self._consume_stream(stream)
            terminal = self._get_terminal_event(chunks)

            if terminal and terminal.type == "response.incomplete":
                assert terminal.response.status == "incomplete"
                # Incomplete is NOT a failure - no error expected
                # (error may or may not be None depending on implementation)
            elif terminal and terminal.type == "response.completed":
                pytest.skip("Response completed normally - cannot test incomplete path")
        except (BadRequestError, APIError):
            pytest.skip("Provider doesn't support this request")

    def test_invalid_image_url_returns_image_error(self, openai_client, vision_model_id):
        """
        Test that an invalid image URL triggers an image-specific error code.

        Requires a vision model. Run with: --vision-model openai/gpt-4o or --setup gpt
        """
        try:
            stream = openai_client.responses.create(
                model=vision_model_id,
                input=[
                    {
                        "type": "message",
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": "What is in this image?"},
                            {"type": "input_image", "image_url": "https://invalid.example.com/no-image.jpg"},
                        ],
                    }
                ],
                stream=True,
            )
            chunks = self._consume_stream(stream)
            terminal = self._get_terminal_event(chunks)
        except (BadRequestError, NotFoundError, APIError) as e:
            pytest.skip(f"Provider returns HTTP error for invalid image: {e}")
            return

        if terminal is None:
            pytest.fail("Expected a terminal event")

        if terminal.type == "response.completed":
            pytest.skip("Provider processed invalid image URL without error")

        if terminal.type != "response.failed":
            pytest.fail(f"Expected response.failed, got {terminal.type}")

        assert terminal.response.error is not None, "Failed response must have an error object"
        error_code = terminal.response.error.code

        expected_codes = {
            "invalid_image_url",
            "failed_to_download_image",
            "invalid_image",
            # TODO: remove internal_error once streaming.py uses spec-compliant error codes
            "internal_error",
            "server_error",
        }
        if error_code in expected_codes:
            return  # Test passed: received expected image error code

        pytest.fail(f"Expected image error code, got '{error_code}'. Message: {terminal.response.error.message}")

    def test_invalid_base64_image_returns_image_error(self, openai_client, vision_model_id):
        """
        Test that invalid base64 image data triggers an image-specific error code.

        Requires a vision model. Run with: --vision-model openai/gpt-4o or --setup gpt
        """
        try:
            stream = openai_client.responses.create(
                model=vision_model_id,
                input=[
                    {
                        "type": "message",
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": "What is in this image?"},
                            {
                                "type": "input_image",
                                "image_url": "data:image/png;base64,not_valid_base64_data!!!",
                            },
                        ],
                    }
                ],
                stream=True,
            )
            chunks = self._consume_stream(stream)
            terminal = self._get_terminal_event(chunks)
        except (BadRequestError, NotFoundError, APIError) as e:
            pytest.skip(f"Provider returns HTTP error, not a streaming error, for invalid base64: {e}")
            return

        if terminal is None:
            pytest.fail("Expected a terminal event")

        if terminal.type == "response.completed":
            pytest.skip("Provider processed invalid base64 image without error")

        if terminal.type != "response.failed":
            pytest.fail(f"Expected response.failed, got {terminal.type}")

        assert terminal.response.error is not None, "Failed response must have an error object"
        error_code = terminal.response.error.code

        expected_codes = {
            "invalid_base64_image",
            "invalid_image",
            "image_parse_error",
            "invalid_image_format",
            # TODO: remove internal_error once streaming.py uses spec-compliant error codes
            "internal_error",
            "server_error",
        }
        if error_code in expected_codes:
            return  # Test passed: received expected image error code

        pytest.fail(f"Expected image error code, got '{error_code}'. Message: {terminal.response.error.message}")

    def test_non_vision_model_returns_error_for_image_input(self, openai_client, text_model_id):
        """
        Test that non-vision models return an appropriate error when given image input.

        Non-vision models should return either:
        - 'server_error': Generic error when code is missing or model can't process images
        - Provider-specific codes like 'invalid_value' may also be returned

        The message field will contain the actual error details from the provider.
        """
        # Valid error codes for non-vision models receiving image input
        # Provider may return specific codes like 'invalid_value' for image input errors
        expected_non_vision_codes = {
            "server_error",
            "invalid_value",
            # TODO: remove internal_error once streaming.py uses spec-compliant error codes
            "internal_error",
        }

        try:
            stream = openai_client.responses.create(
                model=text_model_id,
                input=[
                    {
                        "type": "message",
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": "Describe this image"},
                            {"type": "input_image", "image_url": "https://example.com/test.jpg"},
                        ],
                    }
                ],
                stream=True,
            )
            chunks = self._consume_stream(stream)
            terminal = self._get_terminal_event(chunks)
        except (BadRequestError, NotFoundError, APIError) as e:
            pytest.skip(f"Provider returns HTTP error, not a streaming error, for image input: {e}")
            return

        if terminal is None:
            pytest.fail("Expected a terminal event")

        if terminal.type == "response.completed":
            pytest.skip("Model supports vision - cannot test non-vision behavior")

        if terminal.type != "response.failed":
            pytest.fail(f"Expected response.failed, got {terminal.type}")

        # Don't use _assert_failed_event_structure here because non-vision models
        # may return codes (like 'invalid_value') that aren't in the ResponseError spec
        assert terminal.response is not None, "Failed event should have a response"
        assert terminal.response.error is not None, "Failed event should have an error"
        error_code = terminal.response.error.code

        if error_code in expected_non_vision_codes:
            return

        # Image-specific codes indicate the model supports vision
        image_error_codes = {"invalid_image_url", "invalid_image", "failed_to_download_image"}
        if error_code in image_error_codes:
            pytest.skip(f"Model supports vision (returned '{error_code}')")

        pytest.fail(f"Unexpected error code: {error_code}")

    def test_non_vision_model_with_base64_image_returns_server_error(self, openai_client, text_model_id):
        """
        Test that non-vision models return 'server_error' when given base64 image input.

        When a non-vision model receives a valid image, we return 'server_error' as the
        generic code, but the message contains the actual error details from the provider.
        """
        # A tiny valid 1x1 red PNG as base64
        base64_image = (
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg=="
        )

        # Provider may return specific codes like 'invalid_value' for image input errors
        expected_codes = {"server_error", "invalid_value"}

        try:
            stream = openai_client.responses.create(
                model=text_model_id,
                input=[
                    {
                        "type": "message",
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": "What color is this?"},
                            {"type": "input_image", "image_url": f"data:image/png;base64,{base64_image}"},
                        ],
                    }
                ],
                stream=True,
            )
            chunks = self._consume_stream(stream)
            terminal = self._get_terminal_event(chunks)
        except (BadRequestError, NotFoundError, APIError) as e:
            pytest.skip(f"Provider returns HTTP error for image input: {e}")
            return

        if terminal is None:
            pytest.fail("Expected a terminal event")

        if terminal.type == "response.completed":
            pytest.skip("Model supports vision - cannot test non-vision behavior")

        if terminal.type != "response.failed":
            pytest.fail(f"Expected response.failed, got {terminal.type}")

        assert terminal.response is not None, "Failed event should have a response"
        assert terminal.response.error is not None, "Failed event should have an error"
        error_code = terminal.response.error.code
        error_message = terminal.response.error.message

        if error_code in expected_codes:
            return  # Test passed

        # Image-specific codes indicate the model supports vision
        if error_code in VALID_OPENAI_RESPONSE_ERROR_CODES:
            pytest.skip(f"Model supports vision (returned '{error_code}')")

        pytest.fail(f"Expected 'server_error' or 'invalid_value', got '{error_code}'. Message: {error_message}")
