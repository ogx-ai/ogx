# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""
Error handling and edge case tests for the OGX batch processing functionality.

This module focuses exclusively on testing error conditions, validation failures,
and edge cases for batch operations to ensure robust error handling and graceful
degradation.

Test Categories:
    1. File and Input Validation:
        - test_batch_nonexistent_file_id: Handling invalid file IDs
        - test_batch_malformed_jsonl: Processing malformed JSONL input files
        - test_file_malformed_batch_file: Handling malformed files at upload time
        - test_batch_missing_required_fields: Validation of required request fields

    2. API Endpoint and Model Validation:
        - test_batch_invalid_endpoint: Invalid endpoint handling during creation
        - test_batch_error_handling_invalid_model: Error handling with nonexistent models
        - test_batch_endpoint_mismatch: Validation of endpoint/URL consistency

    3. Batch Lifecycle Error Handling:
        - test_batch_retrieve_nonexistent: Retrieving non-existent batches
        - test_batch_cancel_nonexistent: Cancelling non-existent batches
        - test_batch_cancel_completed: Attempting to cancel completed batches

    4. Parameter and Configuration Validation:
        - test_batch_invalid_completion_window: Invalid completion window values
        - test_batch_invalid_metadata_types: Invalid metadata type validation
        - test_batch_missing_required_body_fields: Validation of required fields in request body

    5. Feature Restriction and Compatibility:
        - test_batch_streaming_not_supported: Streaming request rejection
        - test_batch_mixed_streaming_requests: Mixed streaming/non-streaming validation

Note: Core functionality and OpenAI compatibility tests are located in
test_batches_integration.py for better organization and separation of concerns.

CLEANUP WARNING: These tests create batches to test error conditions but do not
automatically clean them up after test completion. While most error tests create
batches that fail quickly, some may create valid batches that consume resources.
"""

import pytest
from openai import BadRequestError, ConflictError, NotFoundError


class TestBatchesErrorHandling:
    """Error handling and edge case tests for the batches API using OpenAI client."""

    def test_batch_nonexistent_file_id(self, openai_client, batch_helper):
        """Test batch creation with nonexistent input file ID."""

        batch = openai_client.batches.create(
            input_file_id="file-nonexistent-xyz",
            endpoint="/v1/chat/completions",
            completion_window="24h",
        )

        final_batch = batch_helper.wait_for(batch.id, expected_statuses={"failed"})

        # Expecting -
        #  Batch(...,
        #    status='failed',
        #    errors=Errors(data=[
        #      BatchError(
        #        code='invalid_request',
        #        line=None,
        #        message='Cannot find file ..., or organization ... does not have access to it.',
        #        param='file_id')
        #    ], object='list'),
        #    failed_at=1754566971,
        #    ...)

        assert final_batch.status == "failed"
        assert final_batch.errors is not None
        assert len(final_batch.errors.data) == 1
        error = final_batch.errors.data[0]
        assert error.code == "invalid_request"
        assert "cannot find file" in error.message.lower()

    def test_batch_invalid_endpoint(self, openai_client, batch_helper, text_model_id):
        """Test batch creation with invalid endpoint."""
        batch_requests = [
            {
                "custom_id": "invalid-endpoint",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": text_model_id,
                    "messages": [{"role": "user", "content": "Hello"}],
                    "max_tokens": 10,
                },
            }
        ]

        with batch_helper.create_file(batch_requests) as uploaded_file:
            with pytest.raises(BadRequestError) as exc_info:
                openai_client.batches.create(
                    input_file_id=uploaded_file.id,
                    endpoint="/v1/invalid/endpoint",
                    completion_window="24h",
                )

            # Expected -
            #  Error code: 400 - {
            #    'error': {
            #      'detail': "Invalid endpoint: /v1/invalid/endpoint. Supported values: ...",
            #      'code': 'invalid_value',
            #      'param': 'endpoint'
            #    }
            #  }

            error_msg = str(exc_info.value).lower()
            assert exc_info.value.status_code == 400
            assert "invalid" in error_msg
            assert "/v1/invalid/endpoint" in error_msg
            assert "supported values" in error_msg
            assert "endpoint" in error_msg
            assert "invalid_value" in error_msg

    def test_batch_malformed_jsonl(self, openai_client, batch_helper):
        """
        Test batch with malformed JSONL input.

        The /v1/files endpoint requires valid JSONL format, so we provide a well formed line
        before a malformed line to ensure we get to the /v1/batches validation stage.
        """
        with batch_helper.create_file(
            """{"custom_id": "valid", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "test"}}
{invalid json here""",
            "malformed_batch_input.jsonl",
        ) as uploaded_file:
            batch = openai_client.batches.create(
                input_file_id=uploaded_file.id,
                endpoint="/v1/chat/completions",
                completion_window="24h",
            )

            final_batch = batch_helper.wait_for(batch.id, expected_statuses={"failed"})

        # Expecting -
        #  Batch(...,
        #    status='failed',
        #    errors=Errors(data=[
        #      ...,
        #      BatchError(code='invalid_json_line',
        #                 line=2,
        #                 message='This line is not parseable as valid JSON.',
        #                 param=None)
        #    ], object='list'),
        #    ...)

        assert final_batch.status == "failed"
        assert final_batch.errors is not None
        assert len(final_batch.errors.data) > 0
        error = final_batch.errors.data[-1]  # get last error because first may be about the "test" model
        assert error.code == "invalid_json_line"
        assert error.line == 2
        assert "not" in error.message.lower()
        assert "valid json" in error.message.lower()

    @pytest.mark.xfail(reason="Not all file providers validate content")
    @pytest.mark.parametrize("batch_requests", ["", "{malformed json"], ids=["empty", "malformed"])
    def test_file_malformed_batch_file(self, openai_client, batch_helper, batch_requests):
        """Test file upload with malformed content."""

        with pytest.raises(BadRequestError) as exc_info:
            with batch_helper.create_file(batch_requests, "malformed_batch_input_file.jsonl"):
                # /v1/files rejects the file, we don't get to batch creation
                pass

        error_msg = str(exc_info.value).lower()
        assert exc_info.value.status_code == 400
        assert "invalid file format" in error_msg
        assert "jsonl" in error_msg

    def test_batch_retrieve_nonexistent(self, openai_client):
        """Test retrieving nonexistent batch."""
        with pytest.raises(NotFoundError) as exc_info:
            openai_client.batches.retrieve("batch-nonexistent-xyz")

        error_msg = str(exc_info.value).lower()
        assert exc_info.value.status_code == 404
        assert "no batch found" in error_msg or "not found" in error_msg

    def test_batch_cancel_nonexistent(self, openai_client):
        """Test cancelling nonexistent batch."""
        with pytest.raises(NotFoundError) as exc_info:
            openai_client.batches.cancel("batch-nonexistent-xyz")

        error_msg = str(exc_info.value).lower()
        assert exc_info.value.status_code == 404
        assert "no batch found" in error_msg or "not found" in error_msg

    def test_batch_cancel_completed(self, openai_client, batch_helper, text_model_id):
        """Test cancelling already completed batch."""
        batch_requests = [
            {
                "custom_id": "cancel-completed",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": text_model_id,
                    "messages": [{"role": "user", "content": "Quick test"}],
                    "max_tokens": 5,
                },
            }
        ]

        with batch_helper.create_file(batch_requests, "cancel_test_batch_input") as uploaded_file:
            batch = openai_client.batches.create(
                input_file_id=uploaded_file.id,
                endpoint="/v1/chat/completions",
                completion_window="24h",
            )

            final_batch = batch_helper.wait_for(
                batch.id,
                max_wait_time=3 * 60,  # often take 10-11 min, give it 3 min
                expected_statuses={"completed"},
                timeout_action="skip",
            )

        deleted_file = openai_client.files.delete(final_batch.output_file_id)
        assert deleted_file.deleted, f"File {final_batch.output_file_id} was not deleted successfully"

        with pytest.raises(ConflictError) as exc_info:
            openai_client.batches.cancel(batch.id)

        # Expecting -
        #   Error code: 409 - {
        #     'error': {
        #       'message': "Cannot cancel a batch with status 'completed'.",
        #       'type': 'invalid_request_error',
        #       'param': None,
        #       'code': None
        #     }
        #   }
        #
        # NOTE: Same for "failed", cancelling "cancelled" batches is allowed

        error_msg = str(exc_info.value).lower()
        assert exc_info.value.status_code == 409
        assert "cannot cancel" in error_msg

    def test_batch_missing_required_fields(self, openai_client, batch_helper, text_model_id):
        """Test batch with requests missing required fields."""
        batch_requests = [
            {
                # Missing custom_id
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": text_model_id,
                    "messages": [{"role": "user", "content": "No custom_id"}],
                    "max_tokens": 10,
                },
            },
            {
                "custom_id": "no-method",
                "url": "/v1/chat/completions",
                "body": {
                    "model": text_model_id,
                    "messages": [{"role": "user", "content": "No method"}],
                    "max_tokens": 10,
                },
            },
            {
                "custom_id": "no-url",
                "method": "POST",
                "body": {
                    "model": text_model_id,
                    "messages": [{"role": "user", "content": "No URL"}],
                    "max_tokens": 10,
                },
            },
            {
                "custom_id": "no-body",
                "method": "POST",
                "url": "/v1/chat/completions",
            },
        ]

        with batch_helper.create_file(batch_requests, "missing_fields_batch_input") as uploaded_file:
            batch = openai_client.batches.create(
                input_file_id=uploaded_file.id,
                endpoint="/v1/chat/completions",
                completion_window="24h",
            )

            final_batch = batch_helper.wait_for(batch.id, expected_statuses={"failed"})

        # Expecting -
        #  Batch(...,
        #    status='failed',
        #    errors=Errors(
        #      data=[
        #        BatchError(
        #          code='missing_required_parameter',
        #          line=1,
        #          message="Missing required parameter: 'custom_id'.",
        #          param='custom_id'
        #        ),
        #        BatchError(
        #          code='missing_required_parameter',
        #          line=2,
        #          message="Missing required parameter: 'method'.",
        #          param='method'
        #        ),
        #        BatchError(
        #          code='missing_required_parameter',
        #          line=3,
        #          message="Missing required parameter: 'url'.",
        #          param='url'
        #        ),
        #        BatchError(
        #          code='missing_required_parameter',
        #          line=4,
        #          message="Missing required parameter: 'body'.",
        #          param='body'
        #        )
        #    ], object='list'),
        #    failed_at=1754566945,
        #    ...)
        #  )

        assert final_batch.status == "failed"
        assert final_batch.errors is not None
        assert len(final_batch.errors.data) == 4
        no_custom_id_error = final_batch.errors.data[0]
        assert no_custom_id_error.code == "missing_required_parameter"
        assert no_custom_id_error.line == 1
        assert "missing" in no_custom_id_error.message.lower()
        assert "custom_id" in no_custom_id_error.message.lower()
        no_method_error = final_batch.errors.data[1]
        assert no_method_error.code == "missing_required_parameter"
        assert no_method_error.line == 2
        assert "missing" in no_method_error.message.lower()
        assert "method" in no_method_error.message.lower()
        no_url_error = final_batch.errors.data[2]
        assert no_url_error.code == "missing_required_parameter"
        assert no_url_error.line == 3
        assert "missing" in no_url_error.message.lower()
        assert "url" in no_url_error.message.lower()
        no_body_error = final_batch.errors.data[3]
        assert no_body_error.code == "missing_required_parameter"
        assert no_body_error.line == 4
        assert "missing" in no_body_error.message.lower()
        assert "body" in no_body_error.message.lower()

    def test_batch_invalid_completion_window(self, openai_client, batch_helper, text_model_id):
        """Test batch creation with invalid completion window."""
        batch_requests = [
            {
                "custom_id": "invalid-completion-window",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": text_model_id,
                    "messages": [{"role": "user", "content": "Hello"}],
                    "max_tokens": 10,
                },
            }
        ]

        with batch_helper.create_file(batch_requests) as uploaded_file:
            for window in ["1h", "48h", "invalid", ""]:
                with pytest.raises(BadRequestError) as exc_info:
                    openai_client.batches.create(
                        input_file_id=uploaded_file.id,
                        endpoint="/v1/chat/completions",
                        completion_window=window,
                    )
            assert exc_info.value.status_code == 400
            error_msg = str(exc_info.value).lower()
            assert "error" in error_msg
            assert "completion_window" in error_msg

    def test_batch_streaming_not_supported(self, openai_client, batch_helper, text_model_id):
        """Test that streaming responses are not supported in batches."""
        batch_requests = [
            {
                "custom_id": "streaming-test",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": text_model_id,
                    "messages": [{"role": "user", "content": "Hello"}],
                    "max_tokens": 10,
                    "stream": True,  # Not supported
                },
            }
        ]

        with batch_helper.create_file(batch_requests, "streaming_batch_input") as uploaded_file:
            batch = openai_client.batches.create(
                input_file_id=uploaded_file.id,
                endpoint="/v1/chat/completions",
                completion_window="24h",
            )

            final_batch = batch_helper.wait_for(batch.id, expected_statuses={"failed"})

        # Expecting -
        #  Batch(...,
        #    status='failed',
        #    errors=Errors(data=[
        #       BatchError(code='streaming_unsupported',
        #         line=1,
        #         message='Chat Completions: Streaming is not supported in the Batch API.',
        #         param='body.stream')
        #    ], object='list'),
        #    failed_at=1754566965,
        #    ...)

        assert final_batch.status == "failed"
        assert final_batch.errors is not None
        assert len(final_batch.errors.data) == 1
        error = final_batch.errors.data[0]
        assert error.code == "streaming_unsupported"
        assert error.line == 1
        assert "streaming" in error.message.lower()
        assert "not supported" in error.message.lower()
        assert error.param == "body.stream"
        assert final_batch.failed_at is not None

    def test_batch_mixed_streaming_requests(self, openai_client, batch_helper, text_model_id):
        """
        Test batch with mixed streaming and non-streaming requests.

        This is distinct from test_batch_streaming_not_supported, which tests a single
        streaming request, to ensure an otherwise valid batch fails when a single
        streaming request is included.
        """
        batch_requests = [
            {
                "custom_id": "valid-non-streaming-request",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": text_model_id,
                    "messages": [{"role": "user", "content": "Hello without streaming"}],
                    "max_tokens": 10,
                },
            },
            {
                "custom_id": "streaming-request",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": text_model_id,
                    "messages": [{"role": "user", "content": "Hello with streaming"}],
                    "max_tokens": 10,
                    "stream": True,  # Not supported
                },
            },
        ]

        with batch_helper.create_file(batch_requests, "mixed_streaming_batch_input") as uploaded_file:
            batch = openai_client.batches.create(
                input_file_id=uploaded_file.id,
                endpoint="/v1/chat/completions",
                completion_window="24h",
            )

            final_batch = batch_helper.wait_for(batch.id, expected_statuses={"failed"})

        # Expecting -
        #  Batch(...,
        #    status='failed',
        #    errors=Errors(data=[
        #      BatchError(
        #        code='streaming_unsupported',
        #        line=2,
        #        message='Chat Completions: Streaming is not supported in the Batch API.',
        #        param='body.stream')
        #    ], object='list'),
        #    failed_at=1754574442,
        #    ...)

        assert final_batch.status == "failed"
        assert final_batch.errors is not None
        assert len(final_batch.errors.data) == 1
        error = final_batch.errors.data[0]
        assert error.code == "streaming_unsupported"
        assert error.line == 2
        assert "streaming" in error.message.lower()
        assert "not supported" in error.message.lower()
        assert error.param == "body.stream"
        assert final_batch.failed_at is not None

    def test_batch_endpoint_mismatch(self, openai_client, batch_helper, text_model_id):
        """Test batch creation with mismatched endpoint and request URL."""
        batch_requests = [
            {
                "custom_id": "endpoint-mismatch",
                "method": "POST",
                "url": "/v1/embeddings",  # Different from batch endpoint
                "body": {
                    "model": text_model_id,
                    "messages": [{"role": "user", "content": "Hello"}],
                },
            }
        ]

        with batch_helper.create_file(batch_requests, "endpoint_mismatch_batch_input") as uploaded_file:
            batch = openai_client.batches.create(
                input_file_id=uploaded_file.id,
                endpoint="/v1/chat/completions",  # Different from request URL
                completion_window="24h",
            )

            final_batch = batch_helper.wait_for(batch.id, expected_statuses={"failed"})

        # Expecting -
        #  Batch(...,
        #    status='failed',
        #    errors=Errors(data=[
        #      BatchError(
        #        code='invalid_url',
        #        line=1,
        #        message='The URL provided for this request does not match the batch endpoint.',
        #        param='url')
        #    ], object='list'),
        #    failed_at=1754566972,
        #    ...)

        assert final_batch.status == "failed"
        assert final_batch.errors is not None
        assert len(final_batch.errors.data) == 1
        error = final_batch.errors.data[0]
        assert error.line == 1
        assert error.code == "invalid_url"
        assert "does not match" in error.message.lower()
        assert "endpoint" in error.message.lower()
        assert final_batch.failed_at is not None

    def test_batch_error_handling_invalid_model(self, openai_client, batch_helper):
        """Test batch error handling with invalid model."""
        batch_requests = [
            {
                "custom_id": "invalid-model",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": "nonexistent-model-xyz",
                    "messages": [{"role": "user", "content": "Hello"}],
                    "max_tokens": 10,
                },
            }
        ]

        with batch_helper.create_file(batch_requests) as uploaded_file:
            batch = openai_client.batches.create(
                input_file_id=uploaded_file.id,
                endpoint="/v1/chat/completions",
                completion_window="24h",
            )

            final_batch = batch_helper.wait_for(batch.id, expected_statuses={"failed"})

        # Expecting -
        #  Batch(...,
        #    status='failed',
        #    errors=Errors(data=[
        #      BatchError(code='model_not_found',
        #        line=1,
        #        message="The provided model 'nonexistent-model-xyz' is not supported by the Batch API.",
        #        param='body.model')
        #    ], object='list'),
        #    failed_at=1754566978,
        #    ...)

        assert final_batch.status == "failed"
        assert final_batch.errors is not None
        assert len(final_batch.errors.data) == 1
        error = final_batch.errors.data[0]
        assert error.line == 1
        assert error.code == "model_not_found"
        assert "not supported" in error.message.lower()
        assert error.param == "body.model"
        assert final_batch.failed_at is not None

    def test_batch_missing_required_body_fields(self, openai_client, batch_helper, text_model_id):
        """Test batch with requests missing required fields in body (model and messages)."""
        batch_requests = [
            {
                "custom_id": "missing-model",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    # Missing model field
                    "messages": [{"role": "user", "content": "Hello without model"}],
                    "max_tokens": 10,
                },
            },
            {
                "custom_id": "missing-messages",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": text_model_id,
                    # Missing messages field
                    "max_tokens": 10,
                },
            },
        ]

        with batch_helper.create_file(batch_requests, "missing_body_fields_batch_input") as uploaded_file:
            batch = openai_client.batches.create(
                input_file_id=uploaded_file.id,
                endpoint="/v1/chat/completions",
                completion_window="24h",
            )

            final_batch = batch_helper.wait_for(batch.id, expected_statuses={"failed"})

        # Expecting -
        #  Batch(...,
        #    status='failed',
        #    errors=Errors(data=[
        #      BatchError(
        #        code='invalid_request',
        #        line=1,
        #        message='Model parameter is required.',
        #        param='body.model'),
        #      BatchError(
        #        code='invalid_request',
        #        line=2,
        #        message='Messages parameter is required.',
        #        param='body.messages')
        #      ], object='list'),
        #    ...)

        assert final_batch.status == "failed"
        assert final_batch.errors is not None
        assert len(final_batch.errors.data) == 2

        model_error = final_batch.errors.data[0]
        assert model_error.line == 1
        assert "model" in model_error.message.lower()
        assert model_error.param == "body.model"

        messages_error = final_batch.errors.data[1]
        assert messages_error.line == 2
        assert "messages" in messages_error.message.lower()
        assert messages_error.param == "body.messages"

        assert final_batch.failed_at is not None

    def test_batch_invalid_metadata_types(self, openai_client, batch_helper, text_model_id):
        """Test batch creation with invalid metadata types (like lists)."""
        batch_requests = [
            {
                "custom_id": "invalid-metadata-type",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": text_model_id,
                    "messages": [{"role": "user", "content": "Hello"}],
                    "max_tokens": 10,
                },
            }
        ]

        with batch_helper.create_file(batch_requests) as uploaded_file:
            with pytest.raises(Exception) as exc_info:
                openai_client.batches.create(
                    input_file_id=uploaded_file.id,
                    endpoint="/v1/chat/completions",
                    completion_window="24h",
                    metadata={
                        "tags": ["tag1", "tag2"],  # Invalid type, should be a string
                    },
                )

        # Expecting -
        #  Error code: 400 - {'error':
        #    {'message': "Invalid type for 'metadata.tags': expected a string,
        #                 but got an array instead.",
        #     'type': 'invalid_request_error', 'param': 'metadata.tags',
        #     'code': 'invalid_type'}}

        error_msg = str(exc_info.value).lower()
        assert "400" in error_msg
        assert "tags" in error_msg
        assert "string" in error_msg
